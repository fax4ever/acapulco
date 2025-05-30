import torch

from src.models import GNN

class ModelParams:
    def __init__(self, gnn, residual, jk, graph_pooling):
        super().__init__()
        self.gnn = gnn
        self.residual = residual
        self.jk = jk
        self.graph_pooling = graph_pooling
        self.drop_ratio = 0.6
        self.num_layer = 6
        self.emb_dim = 350

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

class MetaModel(torch.nn.Module):
    def __init__(self, num_of_input_models):
        super(MetaModel, self).__init__()
        initial_value = 1 / num_of_input_models
        self.alpha = torch.nn.Parameter(torch.Tensor([initial_value for _ in range(0,6)]))
        self.beta = torch.nn.Parameter(torch.Tensor([initial_value for _ in range(0,6)]))
        self.gamma = torch.nn.Parameter(torch.Tensor([initial_value for _ in range(0,6)]))
        self.delta = torch.nn.Parameter(torch.Tensor([initial_value for _ in range(0,6)]))

    def forward(self, x):
        dim = len(x.size()) - 2
        a, b, c, d = torch.unbind(x, dim)
        return a * self.alpha + b * self.beta + c * self.gamma + d * self.delta

class PredictionDataset(torch.utils.data.Dataset):
    def __init__(self, gcn_predictions, gin_predictions, gcn_bis_predictions, gin_bis_predictions, output_labels):
        super(PredictionDataset).__init__()
        self.gcn_predictions = gcn_predictions
        self.gin_predictions = gin_predictions
        self.gcn_bis_predictions = gcn_bis_predictions
        self.gin_bis_predictions = gin_bis_predictions
        self.output_labels = output_labels

    def __getitem__(self, index):
        x = torch.stack([self.gcn_predictions[index], self.gin_predictions[index], self.gcn_bis_predictions[index], self.gin_bis_predictions[index]])
        y = self.output_labels[index]
        return x, y

    def __len__(self):
        return len(self.gcn_predictions)