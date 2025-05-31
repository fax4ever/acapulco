# Graph Classification Noisy Labels

## Different noises / different strategies

I found this idea probably too late. I should have remembered this concept from the course.
Basically, it seems that the best performances can be achieved only if we differentiate the strategies/models to use for each
data source (A, B, C, D).
While trying to find models that fit all types of noise could lead (according to my trials) to suboptimal results.

What are the differences?

1. `A`: base model + applying a graph mixup data augmentation strategy. This seems to produce decent results for dataset A.
2. `B`: in this case, I took the base model and increased the dropout, layer size, and embedding size.
3. `C`: in this case, I kept the base model, since it already seems to produce good results, for instance in terms of validation set accuracy.
4. `D`: in this case, I added to the base model both the graph mixup data augmentation strategy and a co-teaching/co-learning implementation (to contrast the noise?).

## Common Ideas

Starting from [the GitHub baseline](http://github.com/Graph-Classification-Noisy-Label/hackaton/tree/baselineCe),
I aligned the code with [the Kaggle baseline](https://www.kaggle.com/code/farooqahmadwani/baseline),
specifically to incorporate a validation set and to replace the standard cross-entropy loss with `NoisyCrossEntropyLoss`.

I then trained two models—one GCN and one GIN—that achieved solid accuracy,
and applied an ensemble strategy aimed at outperforming each model individually.
The goal was to leverage their differing error patterns and learn optimal weights for combining their output logits.

To reduce resource consumption, I designed the solution so that each model could be trained independently.
This allows us to retrain a single component without needing to retrain the entire ensemble.

Enforcing determinism by setting `torch.use_deterministic_algorithms(True)` significantly reduced model performance, making it impractical for this task.

For the final solution:

* I added two additional sub-models.
* I implemented a co-teaching/co-learning strategy to address noisy labels.
* I also applied a mixup strategy by adapting code from [https://github.com/ahxt/g-mixup](https://github.com/ahxt/g-mixup).
  * Specifically, to ensure compatibility with our setup, I had to populate both edge and node features:

```python
x = torch.zeros(num_nodes, dtype=torch.int)
edge_attr = torch.normal(0.1, 0.1, size=(edge_index.size(1), 7))
y = torch.ceil(sample_graph_label).to(dtype=torch.int)

pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
```

While not formally proven, several observations emerged during experimentation:

1. Ensembles can outperform individual base models.
2. Greater diversity among models generally leads to stronger ensemble performance.
3. When combining predictions, a linear layer performs significantly worse than summing the Hadamard products of sub-model outputs.
4. `NoisyCrossEntropyLoss` is far more effective than standard cross-entropy loss in the presence of noisy labels.
5. Dropout is a critical regularization technique—especially in noisy settings—and should always be applied.
6. GCN performs best *without* a virtual node and *without* residual connections, using a Jumping Knowledge (JK) strategy with `last` aggregation and `mean` graph pooling.
7. GIN benefits from both a virtual node and residual connections, using JK with `sum` aggregation and `attention` for graph pooling.
8. Training a model on data from one sub-dataset to predict labels in a different sub-dataset generally leads to poor performance and should be avoided.

## Installation and run

```shell
conda create -n deep-learning python=3.11
conda activate deep-learning
pip install -r requirements.txt
```

Optionally if you have an AMD GPU:
```shell
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4/
```

Run training + inference:
```shell
python main.py --test_path /home/fax/bin/data/A/test.json.gz --train_path /home/fax/bin/data/A/train.json.gz
```

Run training (only the metamodel) + inference:
```shell
python main.py --test_path /home/fax/bin/data/A/test.json.gz --train_path /home/fax/bin/data/A/train.json.gz --skip_sub_models_train
```

Run inference only:
```shell
python main.py --test_path /home/fax/bin/data/A/test.json.gz
```

Produce the submission file:
```shell
python src/utils.py
```