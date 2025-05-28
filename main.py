import argparse
import os
import torch
from torch_geometric.loader import DataLoader

from src.hyper_params import ModelParams, MetaModel, PredictionDataset
from src.utils import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from torch.utils.data import random_split
from src.loss import NoisyCrossEntropyLoss
from src.file_util import get_or_create_graph_ds

MY_SEED = 739

# Set the random seed
set_seed(MY_SEED)

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader),  correct / total

def meta_train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        prediction = model(X)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = prediction.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    return total_loss / len(data_loader),  correct / total

def evaluate(data_loader, model, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                total_loss += criterion(output, data.y).item()
            else:
                predictions.extend(pred.cpu().numpy())
    if calculate_accuracy:
        accuracy = correct / total
        return  total_loss / len(data_loader),accuracy
    else:
        return predictions

def meta_evaluate(gcn_outputs, gin_outputs, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for index, gcn_output in enumerate(gcn_outputs):
            gin_output = gin_outputs[index]
            X = torch.stack([gcn_output, gin_output]).to(device)
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


def plot_training_progress(train_losses, train_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()

def training(args, checkpoint_path, checkpoints_folder, criterion, device, logs_folder, model, num_checkpoints,
             optimizer, test_dir_name, train_dataset, val_dataset):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    num_epochs = args.epochs
    best_val_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    # Calculate intervals for saving checkpoints
    if num_checkpoints > 1:
        checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
    else:
        checkpoint_intervals = [num_epochs]
    for epoch in range(num_epochs):
        train_loss, train_acc = train(
            train_loader, model, optimizer, criterion, device,
            save_checkpoints=(epoch + 1 in checkpoint_intervals),
            checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
            current_epoch=epoch
        )

        val_loss, val_acc = evaluate(val_loader, model, device, calculate_accuracy=True)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model updated and saved at {checkpoint_path}")
    # Plot training progress in current directory
    plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))
    plot_training_progress(val_losses, val_accuracies, os.path.join(logs_folder, "plotsVal"))

def meta_training(args, checkpoint_path, checkpoints_folder, criterion, device, logs_folder, model, num_checkpoints,
             optimizer, test_dir_name, train_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Training loop
    num_epochs = args.epochs
    best_accuracy = 0.0
    train_losses = []
    train_accuracies = []

    # Calculate intervals for saving checkpoints
    if num_checkpoints > 1:
        checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
    else:
        checkpoint_intervals = [num_epochs]

    for epoch in range(num_epochs):
        train_loss, train_acc = meta_train(
            train_loader, model, optimizer, criterion, device,
            save_checkpoints=(epoch + 1 in checkpoint_intervals),
            checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
            current_epoch=epoch
        )
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Save logs for training progress
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Save best model
        if train_acc > best_accuracy:
            best_accuracy = train_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model updated and saved at {checkpoint_path}")

    # Plot training progress in current directory
    plot_training_progress(train_losses, train_accuracies, os.path.join(logs_folder, "plots"))

def main(args):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # GCN model
    gcn_model = ModelParams('gcn', False, 'last', 'mean').create_model().to(device)
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=0.001, weight_decay=1e-4)
    gcn_criterion = NoisyCrossEntropyLoss(args.noise_prob)

    # GIN model
    gin_model = ModelParams('gin-virtual', True, 'sum', 'attention').create_model().to(device)
    gin_optimizer = torch.optim.AdamW(gin_model.parameters(), lr=0.005, weight_decay=1e-4)
    gin_criterion = NoisyCrossEntropyLoss(args.noise_prob)

    # Meta model
    meta_model = MetaModel(2).to(device)
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001, weight_decay=1e-4)
    meta_criterion = torch.nn.CrossEntropyLoss() # Maybe NoisyCrossEntropyLoss also here?

    # Get the directory where the main script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 3
    # Identify dataset folder (A, B, C, or D)
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    
    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well

    # Define checkpoint path relative to the script's directory
    gcn_checkpoint_path = os.path.join(script_dir, "checkpoints", f"gcn_{test_dir_name}_best.pth")
    gcn_checkpoints_folder = os.path.join(script_dir, "checkpoints", f"gcn_{test_dir_name}")
    os.makedirs(gcn_checkpoints_folder, exist_ok=True)
    gin_checkpoint_path = os.path.join(script_dir, "checkpoints", f"gin_{test_dir_name}_best.pth")
    gin_checkpoints_folder = os.path.join(script_dir, "checkpoints", f"gin_{test_dir_name}")
    os.makedirs(gin_checkpoints_folder, exist_ok=True)
    meta_checkpoint_path = os.path.join(script_dir, "checkpoints", f"meta_{test_dir_name}_best.pth")
    meta_checkpoints_folder = os.path.join(script_dir, "checkpoints", f"meta_{test_dir_name}")
    os.makedirs(meta_checkpoints_folder, exist_ok=True)

    # Prepare test dataset and loader
    test_dataset = get_or_create_graph_ds('test.bin', args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # If train_path is provided, train the model
    if args.train_path:
        full_dataset = get_or_create_graph_ds('train.bin', args.train_path, transform=add_zeros)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        generator = torch.Generator().manual_seed(MY_SEED)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

        if not args.skip_gcn_train:
            training(args, gcn_checkpoint_path, gcn_checkpoints_folder, gcn_criterion, device, logs_folder, gcn_model, num_checkpoints,
                     gcn_optimizer, test_dir_name, train_dataset, val_dataset)
        else:
            gcn_model.load_state_dict(torch.load(gcn_checkpoint_path))

        if not args.skip_gin_train:
            training(args, gin_checkpoint_path, gin_checkpoints_folder, gin_criterion, device, logs_folder, gin_model, num_checkpoints,
                     gin_optimizer, test_dir_name, train_dataset, val_dataset)
        else:
            gin_model.load_state_dict(torch.load(gin_checkpoint_path))

        if not args.skip_meta_train:
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            gcn_outputs, output_labels = submodel_prediction(val_loader, gcn_model, device)
            gin_outputs, _ = submodel_prediction(val_loader, gin_model, device)
            prediction_dataset = PredictionDataset(gcn_outputs, gin_outputs, output_labels)
            meta_training(args, meta_checkpoint_path, meta_checkpoints_folder, meta_criterion, device, logs_folder, meta_model, num_checkpoints,
                     meta_optimizer, test_dir_name, prediction_dataset)
        else:
            meta_model.load_state_dict(torch.load(meta_checkpoint_path))
    else:
        gcn_model.load_state_dict(torch.load(gcn_checkpoint_path))
        gin_model.load_state_dict(torch.load(gin_checkpoint_path))
        meta_model.load_state_dict(torch.load(meta_checkpoint_path))

    if not args.skip_inference:
        # Generate predictions for the test set using the best model
        gcn_outputs, _ = submodel_prediction(test_loader, gcn_model, device)
        gin_outputs, _ = submodel_prediction(test_loader, gin_model, device)
        predictions = meta_evaluate(gcn_outputs, gin_outputs, meta_model, device)
        save_predictions(predictions, args.test_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument('--skip_gcn_train', type=bool, default=False, help='Avoid to train also the GCN sub model')
    parser.add_argument('--skip_gin_train', type=bool, default=False, help='Avoid to train also the GIN sub model')
    parser.add_argument('--skip_meta_train', type=bool, default=False, help='Avoid to train also the meta model')
    parser.add_argument('--skip_inference', type=bool, default=False, help='Avoid to inference the predictions')
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train (default: 40)')
    parser.add_argument('--noise_prob', type=float, default=0.2, help='Noise probability p in NoisyCrossEntropyLoss')
    args = parser.parse_args()
    main(args)
