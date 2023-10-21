import os


DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoint"
PREDICTION_DIR = "pred"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

TRAIN_FILE = "train.json"
VALID_FILE = "valid.json"
TEST_FILE = "test.json"
CONTEXT_FILE = "context.json"

MC_TRAIN_FILE = "train_mc.json"
MC_VALID_FILE = "valid_mc.json"
MC_TEST_FILE = "test_mc.json"
MC_DATA_FILE = {
    "train": os.path.join(DATA_DIR, MC_TRAIN_FILE),
    "valid": os.path.join(DATA_DIR, MC_VALID_FILE),
}
MC_CONTEXT_NAME = "sent1"
MC_QUESTION_HEADER_NAME = "sent2"
MC_ENDING_LEN = 4
MC_ENDING_NAMES = [f"ending{i}" for i in range(MC_ENDING_LEN)]
MC_LAB_COL_NAME  = "label"
MC_MAX_SEQ_LEN = 512

QA_TRAIN_FILE = "train_qa.json"
QA_VALID_FILE = "valid_qa.json"
QA_TEST_FILE = "test_qa.json"
QA_DATA_FILE = {
    "train": os.path.join(DATA_DIR, QA_TRAIN_FILE),
    "valid": os.path.join(DATA_DIR, QA_VALID_FILE),
}
QA_QUESTION_COL_NAME = "question"
QA_CONTEXT_COL_NAME = "context"
QA_ANS_COL_NAME = "answers"
QA_MAX_SEQ_LEN = 512
