import os
from argparse import Namespace, ArgumentParser

from src.constant import DATA_DIR, TRAIN_FILE, VALID_FILE, TEST_FILE, CONTEXT_FILE
from src.constant import TRAIN_MC_FILE, VALID_MC_FILE, TEST_MC_FILE, TRAIN_QA_FILE, VALID_QA_FILE, TEST_QA_FILE
from src.preprocess import process_mc_data, process_qa_data
from src.utils import load_json, save_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Preprocessing')

    parser.add_argument('--preprocess', type=str, default="mc",
                        help='multiple choice or question answering')
    return parser.parse_args()


if __name__ == "__main__":
    process_fun = {
        "mc": process_mc_data,
        "qa": process_qa_data,
    }

    args = parse_arguments()
    context = load_json(os.path.join(DATA_DIR, CONTEXT_FILE))

    train_data = load_json(os.path.join(DATA_DIR, TRAIN_FILE))
    train_list = [process_fun[args.preprocess](data, context, answer=True) for data in train_data]

    save_json(train_list, os.path.join(DATA_DIR, TRAIN_MC_FILE if args.preprocess=="mc" else TRAIN_QA_FILE))

    valid_data = load_json(os.path.join(DATA_DIR, VALID_FILE))
    valid_list = [process_fun[args.preprocess](data, context, answer=True) for data in valid_data]
    save_json(valid_list, os.path.join(DATA_DIR, VALID_MC_FILE if args.preprocess=="mc" else VALID_QA_FILE))

    test_data = load_json(os.path.join(DATA_DIR, TEST_FILE))
    test_list = [process_fun[args.preprocess](data, context, answer=False) for data in test_data]
    save_json(test_list, os.path.join(DATA_DIR, TEST_MC_FILE if args.preprocess is "mc" else TEST_QA_FILE))
