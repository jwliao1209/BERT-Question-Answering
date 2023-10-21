import os
from argparse import Namespace, ArgumentParser

from src.constants import DATA_DIR, TRAIN_FILE, VALID_FILE, TEST_FILE, CONTEXT_FILE
from src.constants import MC_TRAIN_FILE, MC_VALID_FILE, MC_TEST_FILE, QA_TRAIN_FILE, QA_VALID_FILE, QA_TEST_FILE
from src.utils import load_json, save_json, process_mc_data, process_qa_data


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Preprocessing")

    parser.add_argument("--preprocess", type=str, default="mc",
                        help="multiple choice or question answering")
    parser.add_argument("--inference", action="store_true",
                        help="only for test data")
    parser.add_argument("--test_data", type=str, default=None,
                        help="path of test data")
    parser.add_argument("--context_data", type=str, default=None,
                        help="path of context data")
    return parser.parse_args()


if __name__ == "__main__":
    process_fun = {
        "mc": process_mc_data,
        "qa": process_qa_data,
    }

    args = parse_arguments()

    if args.inference:
        context = load_json(args.context_data)
        test_data = load_json(args.test_data)
        test_list = [process_fun[args.preprocess](data, context, answer=False) for data in test_data]
        save_json(test_list, os.path.join(DATA_DIR, MC_TEST_FILE))
    else:
        context = load_json(os.path.join(DATA_DIR, CONTEXT_FILE))
        train_data = load_json(os.path.join(DATA_DIR, TRAIN_FILE))
        train_list = [process_fun[args.preprocess](data, context, answer=True) for data in train_data]

        save_json(train_list, os.path.join(DATA_DIR, MC_TRAIN_FILE if args.preprocess=="mc" else QA_TRAIN_FILE))

        valid_data = load_json(os.path.join(DATA_DIR, VALID_FILE))
        valid_list = [process_fun[args.preprocess](data, context, answer=True) for data in valid_data]
        save_json(valid_list, os.path.join(DATA_DIR, MC_VALID_FILE if args.preprocess=="mc" else QA_VALID_FILE))

        if args.preprocess == "mc":
            test_data = load_json(os.path.join(DATA_DIR, TEST_FILE))
            test_list = [process_fun[args.preprocess](data, context, answer=False) for data in test_data]
            save_json(test_list, os.path.join(DATA_DIR, MC_TEST_FILE))
