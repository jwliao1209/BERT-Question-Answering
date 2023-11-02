# Question-Answering
This repository is implementation of Homework 1 for CSIE5431 Applied Deep Learning course in 2023 Fall semester at National Taiwan University.


## Setting the Environment
To set the environment, you can run this command:
```
pip install -r configs/requirements.txt
```


## Download dataset and model checkpoint
To download the datasets and model checkpoint, you can run the commad:
```
bash ./download.sh
```

## Reproducing best result
To reproduce our best result, you can run the commad:
```
bash ./run.sh data/context.json data/test.json pred/prediction.csv
```


## Training
### Multiple Choice
To train the multiple choice model, you can run the commad:
```
bash scripts/train_mc.sh
```

### Question Answering
To train the question answering model, you can run the commad:
```
bash scripts/train_qa.sh
```


## Experiment Results
<table>
  <tr>
    <td>Model</td>
    <td>Validation</td>
    <td>Public</td>
    <td>Private</td>
  </tr>
  <tr>
    <td>Not pretrained</td>
    <td>0.0499</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>bert-base-chinese</td>
    <td>0.7983</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>chinese-roberta-wwm-ext-large</td>
    <td>0.8408</td>
    <td>0.8074</td>
    <td>0.8121</td>
  </tr>
<table>


## Operating System and Device
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Acknowledgement
We thank the Hugging Face repository: https://github.com/huggingface/transformers


## Citation
```bibtex
@misc{
    title  = {2023_adl_hw1_question_answering},
    author = {Jia-Wei Liao},
    url    = {https://github.com/jwliao1209/Question-Answering},
    year   = {2023}
}
```
