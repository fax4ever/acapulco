import os
import torch
import logging
import matplotlib.pyplot as plt


def plot_training_progress(train_losses, train_accuracies, output_dir, file_name):
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
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()


class ModelTrainingData:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def append(self, train_loss, train_acc, val_loss, val_acc):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

    def plot_training(self, model_name, output_dir, validation):
        plot_training_progress(self.train_losses, self.train_accuracies, output_dir, f"training_{model_name}.png")
        if validation:
            plot_training_progress(self.val_losses, self.val_accuracies, output_dir, f"validation_{model_name}.png")


class ModelSupport:
    def __init__(self, model, optimizer, criterion, test_dir_name, model_name, base_dir):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.model_name = model_name

        checkpoint_dir = os.path.join(base_dir, "checkpoints", f"{model_name}_{test_dir_name}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.checkpoint_path = os.path.join(checkpoint_dir, f"model_{test_dir_name}")
        self.best_checkpoint_path = os.path.join(base_dir, "checkpoints", f"{model_name}_{test_dir_name}_best.pth")
        self.logs_path = os.path.join(base_dir, "logs", test_dir_name)
        self.training_data = ModelTrainingData()
        self.best_accuracy = 0.0

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def batch(self, X, y):
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item(), output.argmax(dim=1)

    def predict(self, data):
        return self.model(data)

    def save_checkpoint(self, current_epoch):
        checkpoint_file = f"{self.checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(self.model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    def append_epoch_data(self, train_loss, train_acc, val_loss, val_acc, epoch, num_epochs):
        self.training_data.append(train_loss, train_acc, val_loss, val_acc)

        info = f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, " \
               f"Validation Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        logging.info(info)

        # Save best model
        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc
            torch.save(self.model.state_dict(), self.best_checkpoint_path)
            print(f"Best model updated and saved at {self.best_checkpoint_path}")

    def append_epoch_data_no_validation(self, train_loss, train_acc, epoch, num_epochs):
        self.training_data.append(train_loss, train_acc, None, None)

        info = f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
        logging.info(info)

        # Save best model
        if train_acc > self.best_accuracy:
            self.best_accuracy = train_acc
            torch.save(self.model.state_dict(), self.best_checkpoint_path)
            print(f"Best model updated and saved at {self.best_checkpoint_path}")

    def load_best_checkpoint(self):
        self.model.load_state_dict(torch.load(self.best_checkpoint_path))

    def plot_training(self, validation=True):
        # Plot training progress in current directory
        self.training_data.plot_training(self.model_name, self.logs_path, validation)