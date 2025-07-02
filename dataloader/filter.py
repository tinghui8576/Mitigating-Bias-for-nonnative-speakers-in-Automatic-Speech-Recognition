import re
from datasets import Dataset, IterableDataset
from utils.constant import MAX_LABEL_LENGTH, MIN_INPUT_LENGTH, MAX_INPUT_LENGTH, num_proc

def is_in_length_range(length, labels):
    return MIN_INPUT_LENGTH < length < MAX_INPUT_LENGTH and 0 < len(labels) < MAX_LABEL_LENGTH

def filter_audio_length(dataset: Dataset | IterableDataset) -> Dataset | IterableDataset:
    """
    Filter out audio examples or transcript that are not in the length range.
    """
    # Sanity check:
    assert "audio" in dataset.column_names, "Audio column not found in dataset."
    
    dataset = dataset.filter(is_in_length_range,
                            input_columns=["input_length", "labels"],
                            num_proc=num_proc) 
    
    return dataset


def invalid_sentences(example):
    '''
    Given rows with the <> pattern or "IGNORE_TIME_SEGMENT_IN_SCORING" in the sentence, which should be invalid
    '''
    if re.search(r"<[^>]+>", example.get("sentence", "")):
        return False 
    if example.get("sentence", "").startswith("IGNORE_TIME_SEGMENT_IN_SCORING"):
        return False
    return True 

def filter_labels(dataset: Dataset | IterableDataset) -> Dataset | IterableDataset:
    """
    Filter out transcripts that are invalid and those doesn't provide accent information
    """
    # Sanity check:
    assert "sentence" in dataset.column_names, "Sentence column (Transcript) not found in dataset."
    
    dataset = dataset.filter(invalid_sentences, num_proc=num_proc) 
    return dataset

