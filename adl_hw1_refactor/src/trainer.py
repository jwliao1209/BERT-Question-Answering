import os
import torch
from tqdm import tqdm
from src.constants import CHECKPOINT_DIR, QA_MAX_SEQ_LEN
from src.metrics import get_correct_num, get_qa_evalation
from src.postprocess import post_processing_func
from src.utils import dict_to_device, create_and_fill_np_array


class Trainer:
    def __init__(
        self,
        model,
        device,
        train_loader,
        valid_loader,
        optimizer,
        accum_grad_step,
        lr_scheduler,
        *arg,
        **kwarg,
        ):

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_num = len(train_loader)
        self.valid_num = len(valid_loader)
        self.optimizer = optimizer
        self.accum_grad_step = accum_grad_step
        self.lr_scheduler = lr_scheduler

    def train_step(self, batch_data, step):
        NotImplementedError

    def valid_step(self, batch_data, step):
        NotImplementedError

    def train_one_epoch(self):
        self.model.train()
        train_bar = tqdm(self.train_loader, desc=f"Training {self.cur_ep}")
        total_loss = 0
        total_num = 0
        total_correct_number = 0

        for step, batch_data in enumerate(train_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            loss, data_num, correct_number = self.train_step(batch_data, step)

            (loss / self.accum_grad_step).backward()
            if step % self.accum_grad_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

            total_loss += loss.item()
            total_num += data_num
            total_correct_number += correct_number.item()
            train_bar.set_postfix({"loss": total_loss / total_num, "acc": total_correct_number / total_num})
        train_bar.close()
        return

    @torch.no_grad()
    def valid_one_epoch(self):
        self.model.eval()
        valid_bar = tqdm(self.valid_loader, desc=f"Validation {self.cur_ep}")
        total_loss = 0
        total_num = 0
        total_correct_number = 0

        for step, batch_data in enumerate(valid_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            loss, data_num, correct_number = self.valid_step(batch_data, step)
            total_loss += loss
            total_num += data_num
            total_correct_number += correct_number
            valid_bar.set_postfix({"loss": total_loss / total_num, "acc": total_correct_number / total_num})
        valid_bar.close()

        self.model.save_pretrained(
            os.path.join(CHECKPOINT_DIR, f"mc_epoch={self.cur_ep}_acc={total_correct_number / total_num:.4f}")
        )
        return

    def fit(self, epoch):
        self.model.to(self.device)
        for self.cur_ep in range(1, epoch+1):
            self.train_one_epoch()
            self.valid_one_epoch()
        return


class MCTrainer(Trainer):
    def __init__(
        self,
        model,
        device,
        train_loader,
        valid_loader,
        optimizer,
        accum_grad_step,
        lr_scheduler,
        *arg,
        **kwarg,
        ):
        super().__init__(
            model,
            device,
            train_loader,
            valid_loader,
            optimizer,
            accum_grad_step,
            lr_scheduler,
        )

    def train_step(self, batch_data, index):
        outputs = self.model(**batch_data)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        correct_number = get_correct_num(preds, batch_data["labels"])
        return loss, preds.shape[0], correct_number

    def valid_step(self, batch_data, index):
        outputs = self.model(**batch_data)
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)
        correct_number = get_correct_num(preds, batch_data["labels"])
        return loss.item(), preds.shape[0], correct_number.item()


class QATrainer(Trainer):
    def __init__(
        self,
        model,
        device,
        train_loader,
        valid_dataset,
        processed_valid_dataset,
        valid_loader,
        optimizer,
        accum_grad_step,
        lr_scheduler,
        *arg,
        **kwarg,
        ):
        super().__init__(
            model,
            device,
            train_loader,
            valid_loader,
            optimizer,
            accum_grad_step,
            lr_scheduler,
        )
        self.valid_dataset = valid_dataset
        self.processed_valid_dataset = processed_valid_dataset

    def train_step(self, batch_data, index):
        outputs = self.model(**batch_data)
        loss = outputs.loss
        (loss / self.accum_grad_step).backward()

        if index % self.accum_grad_step == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

        return loss.item(), outputs.start_logits.shape[0], 

    def valid_step(self, batch_data, index):
        outputs = self.model(**batch_data)
        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()
        return start_logits, end_logits

    def valid_on_epoch(self):
        self.model.eval()
        start_logits_list, end_logits_list = [], []

        valid_bar = tqdm(self.valid_loader, desc=f"Validation {self.cur_ep}")
        for step, batch_data in enumerate(valid_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            start_logits, end_logits = self.valid_step(batch_data, step)
            start_logits_list.append(start_logits)
            end_logits_list.append(end_logits)
        valid_bar.close()

        start_logits_concat = create_and_fill_np_array(start_logits_list, self.processed_valid_dataset, QA_MAX_SEQ_LEN)
        end_logits_concat = create_and_fill_np_array(end_logits_list, self.processed_valid_dataset, QA_MAX_SEQ_LEN)
        prediction = post_processing_func(
            self.valid_dataset,
            self.processed_valid_dataset,
            (start_logits_concat, end_logits_concat)
        )
        metric = get_qa_evalation()
        eval_metric = metric.compute(
            predictions=prediction.predictions,
            references=prediction.label_ids
        )
        self.model.save_pretrained(
            os.path.join(CHECKPOINT_DIR, f"qa_epoch={self.cur_ep}_acc={eval_metric['exact_match']:.4f}")
        )
        return
