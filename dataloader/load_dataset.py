
from datasets import load_dataset, load_from_disk, Dataset

shared_disk = '/work3/s232855'
def load_process_dataset(dataset_dic: str, **kwargs) -> Dataset:
    """
    Load the dataset and change the coloumn name.
    """
    # Location = shared_disk + "/dataset"
    Location = "/Users/tinghui/.cache/huggingface/datasets"
    # Load dataset
    if dataset_dic["Type"] == "load":
        if dataset_dic.get("Parts"):
            dataset = load_dataset(dataset_dic["Name"], dataset_dic["Parts"], split=dataset_dic["Split"], cache_dir=Location)
        else:
            dataset = load_dataset(dataset_dic["Name"], split=dataset_dic["Split"], cache_dir=Location)
    elif dataset_dic["Type"] == 'disk':
        # dataset = load_from_disk(os.path.join(Location, ds["Name"]))
        dataset = load_from_disk(dataset_dic["Name"])
    else:
        raise ValueError(f"Dataset {dataset_dic["Name"]} not supported")
    
    if dataset_dic["Col_trans"] != "sentence":
        dataset = dataset.rename_column(dataset_dic["Col_trans"], "sentence")
    if dataset_dic["Col_Accent"] != "accents":
        dataset = dataset.rename_column(dataset_dic["Col_Accent"], "accents")
    return dataset 