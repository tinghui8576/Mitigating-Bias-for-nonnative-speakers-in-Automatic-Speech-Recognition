import random
from utils.language_map import language_family_dict, accent_map
from datasets import Dataset, IterableDataset
import utils.constant as const

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
    """
    Balance the dataset by sampling from each group.
    Groups in [7, 14) get low_limit, others get high_limit.
    Sampling is done with probability proportional to 1 / count within each group.
    """
    from collections import Counter
    import numpy as np

    np.random.seed(const.random_state)

    # Build groupings
    grouped_indices = {}
    for idx, item in enumerate(dataset):
        group = item["accent_id"]
        grouped_indices.setdefault(group, []).append(idx)

    selected_indices = []

    for group, indices in grouped_indices.items():
        target = low_limit if group in range(7, 14) else high_limit
        n_available = len(indices)

        # Compute weights: 1/count per index to favor rare items
        # (You can customize this logic if you prefer uniform sampling)
        counts = Counter(indices)
        weights = np.array([1.0 / counts[i] for i in indices])
        probabilities = weights / weights.sum()

        n_sample = min(n_available, target)

        sampled = np.random.choice(
            indices,
            size=n_sample,
            replace=False,
            p=probabilities
        )
        print(f"Group {group}: selected {len(sampled)} / total {n_available}")

        selected_indices.extend(sampled)

    return dataset.select(selected_indices)