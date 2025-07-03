from datasets import DatasetDict
from transformers import WhisperProcessor
from DatasetLoad.preprocessing.preprocessing import preprocess_dataset
train_datalist = [
            {"Name": "edinburghcstr/edacc", "Type": "load", "Split": "validation", "Col_trans": "text", "Col_Accent": "l1", "Limit": [400, 80], 'augment': True},
            # {"Name": "mispeech/speechocean762", "Type": "load", "Split": "train", "Column": "text"},
            # {"Name": "tobiolatunji/afrispeech-200", "Parts": "swahili", "Type": "load", "Split": "train", "Column": "transcript"},
            # {"Name": "tobiolatunji/afrispeech-200", "Parts": "swahili", "Type": "load", "Split": "validation", "Column": "transcript"},
            # {"Name": "tobiolatunji/afrispeech-200", "Parts": "igbo", "Type": "load", "Split": "train", "Column": "transcript"},
            # {"Name": "tobiolatunji/afrispeech-200", "Parts": "igbo", "Type": "load", "Split": "validation", "Column": "transcript"},
            # {"Name": "tobiolatunji/afrispeech-200", "Parts": "yoruba", "Type": "load", "Split": "train", "Column": "transcript"},
            # {"Name": "tobiolatunji/afrispeech-200", "Parts": "yoruba", "Type": "load", "Split": "validation", "Column": "transcript"},
            # {"Name": "l2arctic/shards", "Type": "disk", "Split": "None", "Col_trans": "sentence", "Col_Accent": "accents", "Limit": [4000], 'augment': False},
            # {"Name": "commonvoice/train", "Type": "disk", "Split": "None", "Col_trans": "sentence", "Col_Accent": "accents", "Limit": [2000, 400], 'augment': False},
        ]
eval_datalist = [
            # {"Name": "commonvoice/test", "Type": "disk", "Split": "None", "Col_trans": "sentence", "Col_Accent": "accents"},
            # {"Name": "mispeech/speechocean762", "Type": "load", "Split": "test", "Column": "text"},
            {"Name": "edinburghcstr/edacc", "Type": "load", "Split": "test", "Col_trans": "text", "Col_Accent": "l1", "Limit": None, 'augment': False},
        ]
def load_dataset_dict(processor:WhisperProcessor) -> DatasetDict:

    raw_dataset = DatasetDict()
    raw_dataset["train"] =  preprocess_dataset(train_datalist, processor)
    raw_dataset["eval"] = preprocess_dataset(eval_datalist, processor)

    return raw_dataset
