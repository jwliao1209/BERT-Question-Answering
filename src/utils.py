import json


def load_json(path: str) -> dict:
    with open(path, 'r') as fp:
        obj = json.load(fp)
    return obj


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=4)
    return
