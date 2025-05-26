# Installation and run

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
python main.py --test_path /home/fax/bin/data/A/test.json.gz --train_path /home/fax/bin/data/A/train.json.gz --gnn gcn
```

Run inference only:
```shell
python main.py --test_path /home/fax/bin/data/A/test.json.gz --gnn gcn
```

Produce the submission file:
```shell
python src/utils.py
```