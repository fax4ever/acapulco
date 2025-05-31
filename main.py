import argparse
import os
import torch
from torch_geometric.loader import DataLoader
from src.mix_up_augmentation import mix_up
from src.hyper_params import ModelParams, MetaModel, PredictionDataset
from src.utils import set_seed
import pandas as pd
import logging
from tqdm import tqdm
from torch.utils.data import random_split
from src.loss import NoisyCrossEntropyLoss
from src.file_util import get_or_create_graph_ds
from src.model_support import ModelSupport
from src.co_teaching import co_training
from src.teaching import separate_training

MY_SEED = 739

# Set the random seed
set_seed(MY_SEED)


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def meta_train(model_support: ModelSupport, data_loader, device, save_checkpoints, current_epoch):
    model_support.train()
    total_loss = 0
    correct = 0
    total = 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        total += y.size(0)

        loss, prediction = model_support.batch(X, y)
        total_loss += loss
        correct += (prediction == y).sum().item()

    # Save checkpoints if required
    if save_checkpoints:
        model_support.save_checkpoint(current_epoch)

    return total_loss / len(data_loader), correct / total


def meta_evaluate(gcn_outputs, gin_outputs, gcn_bis_outputs, gin_bis_outputs, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for index, gcn_output in enumerate(gcn_outputs):
            gin_output = gin_outputs[index]
            gcn_bis_output = gcn_bis_outputs[index]
            gin_bis_output = gin_bis_outputs[index]
            X = torch.stack([gcn_output, gin_output, gcn_bis_output, gin_bis_output]).to(device)
            output = model(X)
            pred = output.argmax(dim=0)
            predictions.append(pred.cpu().item())
    return predictions


def submodel_prediction(data_loader, model, device):
    model.eval()
    outputs = []
    labels = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            if data.y is not None:
                labels.extend(data.y)
            outputs.extend(output)
    return outputs, labels


def meta_training(model_support: ModelSupport, train_dataset, batch_size, num_epochs, device, checkpoint_intervals):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

    for epoch in range(num_epochs):
        train_loss, train_acc = meta_train(model_support, train_loader, device,
                                           save_checkpoints=(epoch + 1 in checkpoint_intervals), current_epoch=epoch)
        model_support.append_epoch_data_no_validation(train_loss, train_acc, epoch, num_epochs)

    model_support.plot_training(validation=False)


def save_predictions(predictions, test_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))

    os.makedirs(submission_folder, exist_ok=True)

    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")

    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })

    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


def main(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))

    # GCN model
    gcn_model = ModelParams('gcn', False, 'last', 'mean', test_dir_name).create_model().to(device)
    gcn_model_support = ModelSupport(gcn_model, torch.optim.Adam(gcn_model.parameters(), lr=0.001, weight_decay=1e-4),
                                     NoisyCrossEntropyLoss(args.noise_prob), test_dir_name, 'gcn', script_dir)

    # GIN model
    gin_model = ModelParams('gin-virtual', True, 'sum', 'attention', test_dir_name).create_model().to(device)
    gin_model_support = ModelSupport(gin_model, torch.optim.AdamW(gin_model.parameters(), lr=0.005, weight_decay=1e-4),
                                     NoisyCrossEntropyLoss(args.noise_prob), test_dir_name, 'gin', script_dir)

    # GCN alternative model
    gcn_bis_model = ModelParams('gcn-virtual', False, 'last', 'mean', test_dir_name).create_model().to(device)
    gcn_bis_model_support = ModelSupport(gcn_bis_model,
                                         torch.optim.Adam(gcn_bis_model.parameters(), lr=0.001, weight_decay=1e-4),
                                         NoisyCrossEntropyLoss(args.noise_prob), test_dir_name, 'gcn-bis', script_dir)

    # GIN alternative model
    gin_bis_model = ModelParams('gin', True, 'sum', 'attention', test_dir_name).create_model().to(device)
    gin_bin_model_support = ModelSupport(gin_bis_model,
                                         torch.optim.AdamW(gin_bis_model.parameters(), lr=0.005, weight_decay=1e-4),
                                         NoisyCrossEntropyLoss(args.noise_prob), test_dir_name, 'gin-bis', script_dir)

    # Meta model
    meta_model = MetaModel(4).to(device)
    meta_model_support = ModelSupport(meta_model,
                                      torch.optim.Adam(meta_model.parameters(), lr=0.001, weight_decay=1e-4),
                                      torch.nn.CrossEntropyLoss(), test_dir_name, 'meta', script_dir)

    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well

    # Prepare test dataset and loader
    test_dataset = get_or_create_graph_ds('test.bin', args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Calculate intervals for saving checkpoints
    if args.num_checkpoints > 1:
        checkpoint_intervals = [int((i + 1) * args.epochs / args.num_checkpoints) for i in range(args.num_checkpoints)]
    else:
        checkpoint_intervals = [args.epochs]

    # If train_path is provided, train the model
    if args.train_path:
        full_dataset = get_or_create_graph_ds('train.bin', args.train_path, transform=add_zeros)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(MY_SEED)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        """
        @article{han2022g,
          title={G-Mixup: Graph Data Augmentation for Graph Classification},
          author={Han, Xiaotian and Jiang, Zhimeng and Liu, Ninghao and Hu, Xia},
          journal={arXiv preprint arXiv:2202.07179},
          year={2022}
        }
        """
        if test_dir_name is 'A' or test_dir_name is 'D':
            # Apply mixup only to A and D
            train_dataset = mix_up(train_dataset)

        if args.skip_sub_models_train:
            gcn_model_support.load_best_checkpoint()
            gin_model_support.load_best_checkpoint()
        else:
            # Apply co_training only to D
            if test_dir_name is 'D':
                co_training(gcn_model_support, gin_model_support, train_dataset, val_dataset, args.batch_size,
                            args.epochs, device, checkpoint_intervals)
            else:
                separate_training(gcn_model_support, gin_model_support, train_dataset, val_dataset, args.batch_size,
                                  args.epochs, device, checkpoint_intervals)

        if args.skip_bis_models_train:
            gcn_bis_model_support.load_best_checkpoint()
            gin_bin_model_support.load_best_checkpoint()
        else:
            # Apply co_training only to D
            if test_dir_name is 'D':
                co_training(gcn_bis_model_support, gin_bin_model_support, train_dataset, val_dataset, args.batch_size,
                            args.epochs, device, checkpoint_intervals)
            else:
                separate_training(gcn_bis_model_support, gin_bin_model_support, train_dataset, val_dataset,
                                  args.batch_size, args.epochs, device, checkpoint_intervals)

        if not args.skip_meta_train:
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            gcn_outputs, output_labels = submodel_prediction(val_loader, gcn_model, device)
            gin_outputs, _ = submodel_prediction(val_loader, gin_model, device)
            gcn_bis_outputs, _ = submodel_prediction(val_loader, gcn_bis_model, device)
            gin_bis_outputs, _ = submodel_prediction(val_loader, gin_bis_model, device)
            prediction_dataset = PredictionDataset(gcn_outputs, gin_outputs, gcn_bis_outputs, gin_bis_outputs,
                                                   output_labels)
            meta_training(meta_model_support, prediction_dataset, args.batch_size, args.epochs, device,
                          checkpoint_intervals)
        else:
            meta_model_support.load_best_checkpoint()
    else:
        gcn_model_support.load_best_checkpoint()
        gin_model_support.load_best_checkpoint()
        gcn_bis_model_support.load_best_checkpoint()
        gin_bin_model_support.load_best_checkpoint()
        meta_model_support.load_best_checkpoint()

    if not args.skip_inference:
        # Generate predictions for the test set using the best model
        gcn_outputs, _ = submodel_prediction(test_loader, gcn_model, device)
        gin_outputs, _ = submodel_prediction(test_loader, gin_model, device)
        gcn_bis_outputs, _ = submodel_prediction(test_loader, gcn_bis_model, device)
        gin_bis_outputs, _ = submodel_prediction(test_loader, gin_bis_model, device)
        predictions = meta_evaluate(gcn_outputs, gin_outputs, gcn_bis_outputs, gin_bis_outputs, meta_model, device)
        save_predictions(predictions, args.test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument('--skip_sub_models_train', type=bool, default=False, help='Avoid to train sub models')
    parser.add_argument('--skip_bis_models_train', type=bool, default=False,
                        help='Avoid to train alternative sub models')
    parser.add_argument('--skip_meta_train', type=bool, default=False, help='Avoid to train also the meta model')
    parser.add_argument('--skip_inference', type=bool, default=False, help='Avoid to inference the predictions')
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, default=3, help="Number of checkpoints to save during training.")
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train (default: 40)')
    parser.add_argument('--noise_prob', type=float, default=0.2, help='Noise probability p in NoisyCrossEntropyLoss')
    args = parser.parse_args()
    main(args)
