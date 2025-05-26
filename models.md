# Single models

## Model 1: best GCN atm
0.6565, 0.5286, 0.8275, 0.7515
outcome: {"metric1": 0.7476523897484868}
gnn: gcn
drop_ratio: 0.5
num_layer: 5
emb_dim: 300
batch_size: 32
epochs: 40
noise_prob: 0.2
optimizer: Adam
lr: 0.001
residual: False
JK: last
graph_pooling: mean

## Model 2: best GIN atm
0.6702, 0.5411, 0.8275, 0.7461
outcome: {"metric1": 0.7823301117994464}
gnn: gin-virtual
drop_ratio: 0.5
num_layer: 5
emb_dim: 300
batch_size: 32
epochs: 40
noise_prob: 0.2
optimizer: AdamW
lr: 0.005
residual: True
JK: sum
graph_pooling: attention
