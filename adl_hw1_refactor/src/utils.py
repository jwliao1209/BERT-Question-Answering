import json
import torch
import numpy as np


def load_json(path: str) -> dict:
    with open(path, 'r') as fp:
        obj = json.load(fp)
    return obj


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=4)
    return


def dict_to_device(data: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in data.items()}


def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """
    step = 0
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    for _, output_logit in enumerate(start_or_end_logits):
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size, cols = output_logit.shape
        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat
