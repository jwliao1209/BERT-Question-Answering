from typing import List


def process_mc_data(data: dict, context: List[str], answer: bool = False) -> dict:
    data_mc = {}
    data_mc["id"] = data.get("id", 0)
    data_mc["sent1"] = data.get("question", None)
    data_mc["sent2"] = ""

    for i in range(4):
        data_mc[f"ending{i}"] = context[data.get("paragraphs", None)[i]]
    
    if answer:
        data_mc["label"] = data.get("paragraphs", None).index(data.get("relevant", None))
    
    return data_mc


def process_qa_data(data: dict, context: List[str], answer: bool = False) -> dict:
    data_qa = {}
    data_qa["id"] = data.get("id", 0)
    data_qa["title"] = data.get("id", 0)
    data_qa["context"] = context[data.get("relevant", None)]

    if answer:
        data_qa["answers"] = {
            "text": [data.get("answer", None).get("text", None)],
            "answer_start": [data.get("answer", None).get("start", None)]
        }
    
    return data_qa
