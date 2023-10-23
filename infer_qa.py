import os
import torch

from argparse import Namespace, ArgumentParser
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from transformers import AutoConfig, AutoModelForQuestionAnswering

from src.constants import QA_MAX_SEQ_LEN, PREDICTION_DIR
from src.preprocess import preprocess_valid_qa_func
from src.postprocess import post_processing_func
from src.utils import dict_to_device, create_and_fill_np_array


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Question Answering")

    parser.add_argument("--tokenizer_name", type=str,
                        default="bert-base-chinese",
                        help="tokenizer name")
    parser.add_argument("--checkpoint_folder", type=str,
                        default="checkpoint/qa_epoch=9_acc=83.6823",
                        help="checkpoint folder")
    parser.add_argument("--mc_prediction_path", type=str,
                        default="pred/test_mc_pred.json",
                        help="path of mutiple choice predictions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Prepared datasets
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        trust_remote_code=False,
    )
    datasets = load_dataset("json", data_files={"test": args.mc_prediction_path})
    preprocess_func = partial(preprocess_valid_qa_func, tokenizer=tokenizer)
    processed_test_dataset = datasets["test"].map(
        preprocess_func,
        batched=True,
        remove_columns=datasets["test"].column_names
    )
    test_loader = DataLoader(
        processed_test_dataset.remove_columns(["example_id", "offset_mapping"]),
        batch_size=1,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    # Prepared model
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(args.checkpoint_folder)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.checkpoint_folder,
        config=model_config,
    ).to(device)
    model = model.to(device)
    
    start_logits_list = []
    end_logits_list = []
    inference_bar = tqdm(test_loader, desc=f"Inference")
    for step, batch_data in enumerate(inference_bar, start=1):
        batch_data = dict_to_device(batch_data, device)
        outputs = model(**batch_data)
        start_logits_list.append(outputs.start_logits.detach().cpu().numpy())
        end_logits_list.append(outputs.end_logits.detach().cpu().numpy())

    start_logits_concat = create_and_fill_np_array(start_logits_list, processed_test_dataset, QA_MAX_SEQ_LEN)
    end_logits_concat = create_and_fill_np_array(end_logits_list, processed_test_dataset, QA_MAX_SEQ_LEN)
    prediction = post_processing_func(
        examples=datasets["test"],
        features=processed_test_dataset,
        predictions=(start_logits_concat, end_logits_concat),
        output_dir=os.path.join(PREDICTION_DIR, "test_qa_pred"),
    )
