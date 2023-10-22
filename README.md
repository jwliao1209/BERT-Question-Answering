# Question-Answering

## Environment
```
pip install -r configs/requirements.txt
```

## Download dataset and pretrain weight
```
bash ./download.sh
```

## Reproduce best result
```
bash ./run.sh data/context.json data/test.json pred/prediction.csv
```

## Training
### Multiple Choice
```
bash scripts/train_mc.sh
```

### Question Answering
```
bash scripts/train_qa.sh
```
