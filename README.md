# Graph Classification Noisy Labels

## Ideas

Starting from [the GitHub baseline](http://github.com/Graph-Classification-Noisy-Label/hackaton/tree/baselineCe),
I aligned the code with [the Kaggle baseline](https://www.kaggle.com/code/farooqahmadwani/baseline),
specifically to incorporate a validation set and to replace the standard cross-entropy loss with `NoisyCrossEntropyLoss`.

I then trained two models—one GCN and one GIN—that achieved solid accuracy,
and applied an ensemble strategy aimed at outperforming each model individually.
The goal was to leverage their differing error patterns and learn optimal weights for combining their output logits.

To reduce resource consumption, I designed the solution so that each model could be trained independently.
This allows us to retrain a single component without needing to retrain the entire ensemble.

Enforcing determinism by setting `torch.use_deterministic_algorithms(True)` significantly reduced model performance, making it impractical for this task.

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
python main.py --test_path /home/fax/bin/data/A/test.json.gz --train_path /home/fax/bin/data/A/train.json.gz --skip_gcn true --skip_gin true
```

Run training only on GCN submodel:
```shell
python main.py --test_path /home/fax/bin/data/A/test.json.gz --train_path /home/fax/bin/data/A/train.json.gz --skip_gin true --skip_meta_train true --skip_inference true
```

Run inference only:
```shell
python main.py --test_path /home/fax/bin/data/A/test.json.gz
```

Produce the submission file:
```shell
python src/utils.py
```