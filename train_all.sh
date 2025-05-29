#!/bin/sh

python main.py --test_path /home/fax/bin/data/A/test.json.gz --train_path /home/fax/bin/data/A/train.json.gz
python main.py --test_path /home/fax/bin/data/B/test.json.gz --train_path /home/fax/bin/data/B/train.json.gz
python main.py --test_path /home/fax/bin/data/C/test.json.gz --train_path /home/fax/bin/data/C/train.json.gz
python main.py --test_path /home/fax/bin/data/D/test.json.gz --train_path /home/fax/bin/data/D/train.json.gz
python src/utils.py