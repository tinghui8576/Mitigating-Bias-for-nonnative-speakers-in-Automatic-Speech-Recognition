import random
from utils.language_map import language_family_dict, accent_map
from datasets import Dataset, IterableDataset
def accent_id(batch):
    try:
        batch["accent_id"]= accent_map[language_family_dict [batch["accents"].lower()]]
    except KeyError:
        print(f"The accents {batch["accents"]} is not on the list. Check language_map")
    return batch

def compute_accent_id(dataset: Dataset | IterableDataset) -> Dataset | IterableDataset:
    """
    Compute accent id from accent
    """
    # Sanity check:
    assert "accents" in dataset.column_names, "Accent column not found in dataset."

    dataset = dataset.map(accent_id)
    return dataset

def balance_dataset(dataset, high_limit=300, low_limit=100):
    grouped_indices = {}

    for idx, item in enumerate(dataset):
        group = item["accent_id"]
        grouped_indices.setdefault(group, []).append(idx)

    selected_indices = []
    for group, indices in grouped_indices.items():
        target = low_limit if group in range(7, 14) else high_limit
        random.seed(42)
        sample = random.sample(indices, min(len(indices), target))
        print(f"Group {group}: selected {len(sample)} / total {len(indices)}")
        selected_indices.extend(sample)

    # return dataset.select(selected_indices)
    return dataset.select([10])