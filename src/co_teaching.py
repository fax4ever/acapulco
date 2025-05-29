import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from src.model_support import ModelSupport
import torch.nn.functional as F


def evaluate(model_support: ModelSupport, data_loader, device):
    model_support.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model_support.predict(data)
            prediction = output.argmax(dim=1)

            correct += (prediction == data.y).sum().item()
            total += data.y.size(0)
            total_loss += criterion(output, data.y).item()

    return total_loss / len(data_loader), correct / total


def train(model_support_1: ModelSupport, model_support_2: ModelSupport, data_loader, device, save_checkpoints,
          current_epoch, forget_rate=0.2):
    model_support_1.train()
    model_support_2.train()
    total_loss = [0, 0]
    correct = [0, 0]
    total = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        total += data.y.size(0)

        out1 = model_support_1.model(data)
        out2 = model_support_2.model(data)
        loss1 = F.cross_entropy(out1, data.y, reduction='none')
        loss2 = F.cross_entropy(out2, data.y, reduction='none')

        # Select small-loss instances
        idx1 = torch.argsort(loss1)[:int((1 - forget_rate) * len(loss1))]
        idx2 = torch.argsort(loss2)[:int((1 - forget_rate) * len(loss2))]

        # Cross-update
        final_loss1 = model_support_1.criterion(out1[idx2], data.y[idx2])
        final_loss2 = model_support_2.criterion(out2[idx1], data.y[idx1])

        model_support_1.optimizer.zero_grad()
        final_loss1.backward()
        model_support_1.optimizer.step()

        model_support_2.optimizer.zero_grad()
        final_loss2.backward()
        model_support_2.optimizer.step()

        total_loss[0] += final_loss1.item()
        total_loss[1] += final_loss2.item()
        prediction_1 = out1.argmax(dim=1)
        prediction_2 = out2.argmax(dim=1)
        correct[0] += (prediction_1 == data.y).sum().item()
        correct[1] += (prediction_2 == data.y).sum().item()

    # Save checkpoints if required
    if save_checkpoints:
        model_support_1.save_checkpoint(current_epoch)
        model_support_2.save_checkpoint(current_epoch)

    return total_loss[0] / len(data_loader), correct[0] / total, total_loss[1] / len(data_loader), correct[1] / total


def co_training(model_support_1: ModelSupport, model_support_2: ModelSupport, train_dataset, val_dataset, batch_size,
                num_epochs, device, checkpoint_intervals):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        train_loss_1, train_acc_1, train_loss_2, train_loss_2 = train(model_support_1, model_support_2, train_loader, device,
                                      save_checkpoints=(epoch + 1 in checkpoint_intervals), current_epoch=epoch)
        val_loss, val_acc = evaluate(model_support_1, val_loader, device)
        model_support_1.append_epoch_data(train_loss_1, train_acc_1, val_loss, val_acc, epoch, num_epochs)
        val_loss, val_acc = evaluate(model_support_2, val_loader, device)
        model_support_2.append_epoch_data(train_loss_2, train_loss_2, val_loss, val_acc, epoch, num_epochs)

    model_support_1.plot_training()