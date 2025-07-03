from functools import partial
from typing import Iterable
from datasets import DatasetDict


from Refer.utils.constants import DEFAULT_LABEL_STR_COL
from datasets import load_dataset, load_from_disk, concatenate_datasets, Audio

def load_dataset_dict(dataset_name: str, **kwargs) -> DatasetDict:
    # """Load the dataset dictionary."""
    # if dataset_name in TRAIN_DATASET_NAME_TO_LOAD_FCT:
    #     dataset_dict = TRAIN_DATASET_NAME_TO_LOAD_FCT[dataset_name](**kwargs)
    # else:
    #     raise ValueError(f"Dataset {dataset_name} not supported")
    dataset = load_from_disk(dataset_name)
    return dataset


def gen_from_dataset(dataset) -> Iterable[dict]:
    """Yield the audio and reference from the dataset."""
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": item[DEFAULT_LABEL_STR_COL]}