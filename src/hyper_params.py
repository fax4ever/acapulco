import numpy as np
import torch
from torch import tensor

from src.models import GNN

class ModelParams:
    def __init__(self, gnn, residual, jk, graph_pooling):
        super().__init__()
        self.gnn = gnn
        self.residual = residual
        self.jk = jk
        self.graph_pooling = graph_pooling
        self.drop_ratio = 0.5
        self.num_layer = 5
        self.emb_dim = 300

    def create_model(self):
        if self.gnn == 'gin':
            gnn_type='gin'
            virtual_node=False
        elif self.gnn == 'gin-virtual':
            gnn_type='gin'
            virtual_node=True
        elif self.gnn == 'gcn':
            gnn_type='gcn'
            virtual_node=False
        elif self.gnn == 'gcn-virtual':
            gnn_type='gcn'
            virtual_node=True
        else:
            raise ValueError('Invalid GNN type')
        return GNN(gnn_type=gnn_type, num_class=6, num_layer=self.num_layer, emb_dim=self.emb_dim,
                   drop_ratio=self.drop_ratio, virtual_node=virtual_node, residual=self.residual, JK=self.jk,
                   graph_pooling=self.graph_pooling)

class TrainingParams:
    def __init__(self, optimizer):
        super().__init__()
        self.optimizer = optimizer # TODO use it for training
        self.batch_size = 32 # TODO use it for training
        self.epochs = 40 # TODO use it for training
        self.noise_prob = 0.2 # TODO use it for training

class MetaModel(torch.nn.Module):
    def __init__(self, num_of_input_models):
        super(MetaModel, self).__init__()
        initial_value = 1 / num_of_input_models
        self.alpha = torch.nn.Parameter(torch.Tensor([initial_value for _ in range(0,6)]))
        self.beta = torch.nn.Parameter(torch.Tensor([initial_value for _ in range(0,6)]))

    def forward(self, x):
        dim = len(x.size()) - 2
        a, b = torch.unbind(x, dim)
        return a * self.alpha + b * self.beta

class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, gcn_predictions, gin_predictions, output_labels):
        super(PredictionDataset).__init__()
        self.gcn_predictions = gcn_predictions # to apply the
        self.gin_predictions = gin_predictions
        self.output_labels = output_labels

    def __getitem__(self, index):
        x = torch.stack([self.gcn_predictions[index], self.gin_predictions[index]])
        y = self.output_labels[index]
        return x, y

    def __len__(self):
        return len(self.gcn_predictions)