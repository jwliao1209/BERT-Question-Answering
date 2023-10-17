import numpy as np
import evaluate


def get_correct_num(y_pred: np.array, y_true: np.array) -> np.array:
    return (y_pred == y_true).float().sum()


def get_qa_evalation():
    return evaluate.load("squad")
