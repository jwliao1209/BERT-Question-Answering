import os
import torch

from argparse import Namespace, ArgumentParser
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from transformers import AutoConfig, AutoModelForMultipleChoice

from src.constants import DATA_DIR, MC_TEST_FILE, PREDICTION_DIR
from src.preprocess import preprocess_mc_func
from src.utils import dict_to_device, save_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Mutiple Choice")

    parser.add_argument("--tokenizer_name", type=str,
                        default="bert-base-chinese",
                        help="tokenizer name")
    parser.add_argument("--checkpoint_folder", type=str,
                        default="checkpoint/mutiple_choice_epoch=5_acc=0.9674",
                        help="checkpoint folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Prepared datasets
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        trust_remote_code=False,
    )
    datasets = load_dataset("json", data_files={"test": os.path.join(DATA_DIR, MC_TEST_FILE)})
    preprocess_func = partial(preprocess_mc_func, tokenizer=tokenizer, train=False)
    processed_test_dataset = datasets["test"].map(
        preprocess_func,
        batched=True,
        remove_columns=datasets["test"].column_names
    )
    test_loader = DataLoader(
        processed_test_dataset,
        batch_size=2,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    # Prepared model
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(args.checkpoint_folder)
    model = AutoModelForMultipleChoice.from_pretrained(
        args.checkpoint_folder,
        config=model_config,
    ).to(device)
    model = model.to(device)
    model.eval()
    
    pred_list = []
    inference_bar = tqdm(test_loader, desc=f"Inference")
    for _, batch_data in enumerate(inference_bar, start=1):
        with torch.no_grad():
            batch_data = dict_to_device(batch_data, device)
            outputs = model(**batch_data)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy().tolist()
            pred_list += preds

    test_data_list = []
    for index, pred in enumerate(pred_list):
        test_data = datasets["test"][index]
        test_data_list.append(
            {
                "id": test_data['id'],
                "context": test_data[f'ending{pred}'],
                "question": test_data['sent1'],
                "answers": {
                    "text": [test_data[f"ending{pred}"][0]],
                    "answer_start": [0]
                }
            }
        )
    save_json(test_data_list, os.path.join(PREDICTION_DIR, "test_mc_pred.json"))
