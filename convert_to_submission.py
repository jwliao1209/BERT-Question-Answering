import pandas as pd
from argparse import Namespace, ArgumentParser
from src.utils import load_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Generate Submission')

    parser.add_argument('--prediction_path', type=str,
                        default="prediction/test_qa_pred/eval_predictions.json",
                        help='prediction path')
    parser.add_argument('--output_path', type=str,
                        default="prediction/submission.csv",
                        help='output path')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    pred = load_json(args.prediction_path)
    pred_df = pd.DataFrame(pred.items(), columns=["id", "answer"])
    pred_df.to_csv(args.output_path, index=False)
