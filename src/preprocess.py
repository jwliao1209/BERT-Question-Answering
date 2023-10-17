from itertools import chain
from transformers import AutoTokenizer

from src.constants import MC_CONTEXT_NAME, MC_QUESTION_HEADER_NAME, MC_ENDING_LEN, MC_ENDING_NAMES, MC_LAB_COL_NAME, MC_MAX_SEQ_LEN
from src.constants import QA_MAX_SEQ_LEN, QA_QUESTION_COL_NAME, QA_CONTEXT_COL_NAME, QA_ANS_COL_NAME

def flatten_list(input_list: list) -> list:
    return list(chain(*input_list))


def unflatten_list(input_list: list, sub_list_num) -> list:
    return [
        input_list[i : i + sub_list_num]
        for i in range(0, len(input_list), sub_list_num)
    ]


def preprocess_mc_func(data: dict, tokenizer: AutoTokenizer, train=True) -> dict:
    first_sentences = [[context] * MC_ENDING_LEN for context in data[MC_CONTEXT_NAME]]
    question_headers = data[MC_QUESTION_HEADER_NAME]
    second_sentences = [
        [f"{header} {data[end][i]}" for end in MC_ENDING_NAMES]
        for i, header in enumerate(question_headers)
    ]

    tokenized_data = tokenizer(
        flatten_list(first_sentences),
        flatten_list(second_sentences),
        max_length=MC_MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
    )
    tokenized_data = {k: unflatten_list(v, MC_ENDING_LEN) for k, v in tokenized_data.items()}

    if train:
        tokenized_data["labels"] = data[MC_LAB_COL_NAME]

    return tokenized_data


def preprocess_train_qa_func(data: dict, tokenizer: AutoTokenizer) -> dict:
    pad_on_right = tokenizer.padding_side == "right"
    data[QA_QUESTION_COL_NAME] = [q.lstrip() for q in data[QA_QUESTION_COL_NAME]]

    tokenized_data = tokenizer(
        data[QA_QUESTION_COL_NAME if pad_on_right else QA_CONTEXT_COL_NAME],
        data[QA_CONTEXT_COL_NAME if pad_on_right else QA_QUESTION_COL_NAME],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=QA_MAX_SEQ_LEN,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_data.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_data.pop("offset_mapping")

    tokenized_data["start_positions"] = []
    tokenized_data["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_data["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_data.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = data[QA_ANS_COL_NAME][sample_index]

        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_data["start_positions"].append(cls_index)
            tokenized_data["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_data["start_positions"].append(cls_index)
                tokenized_data["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_data["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_data["end_positions"].append(token_end_index + 1)

    return tokenized_data


def preprocess_valid_qa_func(data: dict, tokenizer: AutoTokenizer) -> dict:
    pad_on_right = tokenizer.padding_side == "right"
    data[QA_QUESTION_COL_NAME] = [q.lstrip() for q in data[QA_QUESTION_COL_NAME]]

    tokenized_data = tokenizer(
        data[QA_QUESTION_COL_NAME if pad_on_right else QA_CONTEXT_COL_NAME],
        data[QA_CONTEXT_COL_NAME if pad_on_right else QA_QUESTION_COL_NAME],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=QA_MAX_SEQ_LEN,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_data.pop("overflow_to_sample_mapping")

    tokenized_data["example_id"] = []
    for i in range(len(tokenized_data["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_data.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_data["example_id"].append(data["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_data["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_data["offset_mapping"][i])
        ]

    return tokenized_data
