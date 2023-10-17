import math
import torch
import wandb

from argparse import Namespace, ArgumentParser
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from transformers import AutoConfig, AutoModelForQuestionAnswering
from transformers import get_scheduler

from src.constants import QA_DATA_FILE
from src.preprocess import preprocess_train_qa_func, preprocess_valid_qa_func
from src.optimizer import get_optimizer
from src.trainer import QATrainer
from src.utils import set_random_seeds


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Question Answering")

    parser.add_argument("--tokenizer_name", type=str,
                        default="bert-base-chinese",
                        help="tokenizer name")
    parser.add_argument("--model_name_or_path", type=str,
                        default="hfl/chinese-roberta-wwm-ext-large",
                        help="model name or path")
    parser.add_argument("--batch_size", type=int,
                        default=10,
                        help="batch size")
    parser.add_argument("--accum_grad_step", type=int,
                        default=8,
                        help="accumulation gradient steps")
    parser.add_argument("--epoch", type=int,
                        default=10,
                        help="number of epochs")
    parser.add_argument("--lr", type=float,
                        default=3e-5,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=0,
                        help="weight decay")
    parser.add_argument("--lr_scheduler", type=str,
                        default="cosine",
                        help="learning rate scheduler")
    parser.add_argument("--warm_up_step", type=int,
                        default=300,
                        help="number of warm up steps")
    parser.add_argument("--device_id", type=int,
                        default=0,
                        help="deivce id")

    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()

    # Prepared datasets
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        trust_remote_code=False,
    )
    datasets = load_dataset("json", data_files=QA_DATA_FILE)

    preprocess_train_func = partial(preprocess_train_qa_func, tokenizer=tokenizer)
    processed_train_dataset = datasets["train"].map(
        preprocess_train_func,
        batched=True,
        remove_columns=datasets["train"].column_names
    )
    train_loader = DataLoader(
        processed_train_dataset,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
    )

    preprocess_valid_func = partial(preprocess_valid_qa_func, tokenizer=tokenizer)
    processed_valid_dataset = datasets["valid"].map(
        preprocess_valid_func,
        batched=True,
        remove_columns=datasets["valid"].column_names
    )
    valid_loader = DataLoader(
        processed_valid_dataset.remove_columns(["example_id", "offset_mapping"]),
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        shuffle=False,
    )

    # Prepared model
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        config=model_config,
    ).to(device)

    # Prepared optimizer and learning rate scheduler
    optimizer = get_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay
    )
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.accum_grad_step)
    max_train_steps = args.epoch * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warm_up_step * args.accum_grad_step,
        num_training_steps=max_train_steps * args.accum_grad_step,
    )
    # Prepared logger
    wandb.init(
        project="adl_hw1", 
        name="experiment_qa", 
        config={
            "tokenizer": args.tokenizer_name,
            "model": args.model_name_or_path,
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "accum_grad_step": args.accum_grad_step,
            "optimizer": "adamw",
            "lr_scheduler": args.lr_scheduler,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "num_warmup_steps": args.warm_up_step,
        }
    )

    trainer = QATrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_dataset=datasets["valid"],
        processed_valid_dataset=processed_valid_dataset,
        optimizer=optimizer,
        accum_grad_step=args.accum_grad_step,
        lr_scheduler=lr_scheduler,
        logger=wandb,
    )
    trainer.fit(epoch=args.epoch)
    wandb.finish()
