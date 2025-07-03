from typing import Dict
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from torchmetrics.text import WordErrorRate, WordInfoLost
from transformers import WhisperProcessor, EvalPrediction

WER = WordErrorRate()
WIL = WordInfoLost()
normalizer = BasicTextNormalizer()

def compute_metrics_fct(pred: EvalPrediction,
                    processor: WhisperProcessor)-> Dict[str, float]:
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [normalizer(pred) for pred in pred_str]
    label_str = [normalizer(label) for label in label_str]

    wer = WER(pred_str, label_str).item() * 100
    return {"wer": wer}  
