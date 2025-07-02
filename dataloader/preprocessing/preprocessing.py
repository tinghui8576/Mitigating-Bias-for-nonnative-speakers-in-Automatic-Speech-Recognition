import torch
from typing import Any, Dict
from functools import partial
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperProcessor
from dataloader.preprocessing.augmentation import augment_audio
from dataloader.filter import filter_audio_length, filter_labels
from dataloader.utils import compute_accent_id, balance_dataset
from datasets import Audio, Dataset, concatenate_datasets
from utils.constant import num_proc

do_lower_case = True
do_remove_punctuation = True
normalizer = BasicTextNormalizer()

def prepare_batch(batch: Dict[str, Any], 
                  processor: WhisperProcessor)-> Dict[str, Any]:
    """
    Extract features from audio 
    Normalize and tokenize the transcript 

    ! Whisper have the requirement that audio should be in 16kHZ
    """  
    audio = batch["audio"]

    if not isinstance(audio, dict) or "array" not in audio or audio["array"] is None:
        print("⚠️ Problem with audio input:", audio)
        return {}

    if torch.isnan(torch.tensor(audio["array"])).any():
        print("⚠️ Found NaNs in audio array")

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    
    transcription = batch["sentence"]
    
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    if transcription is None or len(transcription.strip()) == 0:
        print("⚠️ Empty transcription:", transcription)

    try:
        # encode target text to label ids
        batch["labels"] = processor.tokenizer(transcription).input_ids
        
    except Exception as e:
        print("⚠️ Error tokenizing:", transcription, "\n", e)
        batch["labels"] = []
    return batch

from dataloader.load_dataset import load_process_dataset

def preprocess_dataset(datalist:Dict[str, Any],
                       processor: WhisperProcessor)-> Dataset:
    processed_datasets = []
    for ds in datalist:
        print(f"Loading: {ds['Name']} ({ds['Split']})")
        dataset = load_process_dataset(ds)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

        dataset = filter_labels(dataset)  
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["audio", "sentence", "accents"]])
        dataset = compute_accent_id(dataset)

        dataset = balance_dataset(dataset, high_limit = ds['High_Limit'], low_limit = ds['Low_Limit'])
        if ds['augment']:
            augment_dataset = partial(augment_audio, sample_rate=processor.feature_extractor.sampling_rate)
            dataset = dataset.map(augment_dataset, num_proc=num_proc)
            print('augment')
            
        prepare_dataset = partial(prepare_batch,
                                  processor=processor)
        dataset = dataset.map(prepare_dataset, num_proc=num_proc)
        
        dataset = filter_audio_length(dataset)
       
        processed_datasets.append(dataset)
    return concatenate_datasets(processed_datasets).shuffle(seed=22)
