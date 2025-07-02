import torch
from utils.Model import Whisper
from torchmetrics.text import WordErrorRate, WordInfoLost
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict
from transformers import get_scheduler, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator
from torch.utils.data import DataLoader
import wandb
import argparse
import utils.load as load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
#######################     ARGUMENT PARSING        #########################
def parse_args():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Train a speech classification model with wandb logging.")
    # Define arguments
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=False, 
        default='openai/whisper-small', 
        help='Huggingface model name to fine-tune. Eg: openai/whisper-small'
    )
    parser.add_argument(
        '--language', 
        type=str, 
        required=False, 
        default='English', 
        help='Language the model is being adapted to in Camel case.'
    )
    parser.add_argument(
        '--sampling_rate', 
        type=int, 
        required=False, 
        default=16000, 
        help='Sampling rate of audios.'
    )
    parser.add_argument(
        '--num_proc', 
        type=int, 
        required=False, 
        default=2, 
        help='Number of parallel jobs to run. Helps parallelize the dataset prep stage.'
    )
    parser.add_argument(
        '--train_strategy', 
        type=str, 
        required=False, 
        default='steps', 
        help='Training strategy. Choose between steps and epoch.'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        required=False, 
        default=1.75e-5, 
        help='Learning rate for the fine-tuning process.'
    )
    parser.add_argument(
        '--warmup', 
        type=int, 
        required=False, 
        default=20000, 
        help='Number of warmup steps.'
    )
    parser.add_argument(
        '--train_batchsize', 
        type=int, 
        required=False, 
        default=48, 
        help='Batch size during the training phase.'
    )
    parser.add_argument(
        '--eval_batchsize', 
        type=int, 
        required=False, 
        default=32, 
        help='Batch size during the evaluation phase.'
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        required=False, 
        default=20, 
        help='Number of epochs to train for.'
    )
    parser.add_argument(
        '--num_steps', 
        type=int, 
        required=False, 
        default=100000, 
        help='Number of steps to train for.'
    )
    parser.add_argument(
        '--resume_from_ckpt', 
        type=str, 
        required=False, 
        default=None, 
        help='Path to a trained checkpoint to resume training from.'
    )
    parser.add_argument(
        "--save_model", 
        type=str, 
        choices=['True', 'False'], 
        default='True', 
        help="Store the training model or no")
    parser.add_argument(
        "--project_name", 
        type=str, 
        default='None', 
        help="Name of the project")
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=False, 
        default='output_model_dir', 
        help='Output directory for the checkpoints generated.'
    )
    parser.add_argument('--sens_name', default='Gender', choices=['Gender', 'Age'])
    
    parser.add_argument('--use_cuda', action='store_true', help="Flag to use GPU if available.")
    # parser.add_argument('--wandb', type=str, choices=['True', 'False'], default='True', help="Use wandb logging (True or False).")
    
    
    return parser.parse_args()

opt = vars(parse_args())
# Check if a GPU is available
opt['device'] = torch.device('cuda' if opt['use_cuda'] and torch.cuda.is_available() else 'cpu')

gradient_checkpointing = True


do_normalize_eval = True
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()

wandb.login()
import time
run_name = f"whisper_finetune_{int(time.time())}"
wandb.init(project="whisper-finetune", config=opt, name=run_name)

############################        DATASET LOADING AND PREP        ##########################
processor = WhisperProcessor.from_pretrained(opt["model_name"], language=opt["language"], task="transcribe")


raw_dataset = DatasetDict()
raw_dataset["train"] = load.load_all_datasets(opt, processor, 'train')
raw_dataset["eval"] = load.load_all_datasets(opt, processor, 'eval')

# print(raw_dataset)
###############################     DATA COLLATOR AND METRIC DEFINITION     ########################

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        # print(len(input_features))
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # print(f"Received {len(features)} features")
        # for i, feature in enumerate(features):
        #     print(f"Feature {i} keys: {list(feature.keys())}")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        accent_features = [feature["accent_id"] for feature in features]
        # accent_features = [{"accent_id": feature["accent_id"]} for feature in features]
        # Process accent IDs and convert to tensor
        if isinstance(accent_features[0], torch.Tensor):
            accents = torch.stack(accent_features)
        else:
            accents = torch.tensor(accent_features, dtype=torch.long)

        # Add accents to the batch
        batch["accent_id"] = accents
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

print('DATASET PREPARATION COMPLETED')

WER = WordErrorRate()
WIL = WordInfoLost()
# from time import perf_counter
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=opt['language'], task="transcribe")
    # Start = perf_counter()
    for batch in dataloader:
        input_features = batch["input_features"]
        label_ids = batch["labels"]
        accent_id = batch["accent_id"]

        # generated_ids = model.module.generate(input_features=input_features)
        generated_ids = model.module.generate(
                input_features=input_features,
                accent_id=accent_id,
                forced_decoder_ids=forced_decoder_ids
            )
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        preds = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        
        all_preds.extend(preds)
        all_labels.extend(refs)
        # print(perf_counter()-Start)

    if do_normalize_eval:
        all_preds = [normalizer(pred) for pred in all_preds]
        all_labels = [normalizer(label) for label in all_labels]
    
    wer = WER(all_preds, all_labels).item() * 100
    return wer

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    wer = WER(pred_str, label_str).item() * 100
    return {"wer": wer}    


#############################       MODEL LOADING       #####################################
model = Whisper(model_name=opt['model_name'], language=opt['language'], device=opt['device'])
model.train()

# Ensure LoRA layers have gradients enabled
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# lora.mark_only_lora_as_trainable(model)
###############################     TRAINING ARGS AND TRAINING      ############################
# print(raw_dataset["train"][0])

training_args = dict(
        output_dir=opt['output_dir'],
        per_device_train_batch_size=opt['train_batchsize'],
        gradient_accumulation_steps=2,
        learning_rate=opt['learning_rate'],
        warmup_steps=opt['warmup'],
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        # bf16=True,
        per_device_eval_batch_size=opt['eval_batchsize'],
        predict_with_generate=True,
        generation_max_length=300,
        num_train_epochs = 1,

        evaluation_strategy="steps",
        # eval_strategy="steps",
        eval_steps=50, 
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        save_strategy="steps",
        save_steps=50, 
        max_steps=opt['num_steps'],
        save_total_limit=10,
        weight_decay=0.01,
        max_grad_norm=0.5,

        logging_strategy="steps",
        logging_steps=50, 
        report_to=["wandb", "tensorboard"],
        
        optim="adamw_bnb_8bit",
        # optim="adamw_torch",
        resume_from_checkpoint=opt['resume_from_ckpt'],
    )

from Refer.trainer.ewc_finetuning import EWCFinetuningTrainer, EWCFinetuningTrainingArguments
from Refer.utils.ewc_finetune_config import EWCFinetuneConfig
config = EWCFinetuneConfig.from_yaml("Refer/config/test.yaml")
training_args = EWCFinetuningTrainingArguments(dirpath_ewc=config.dirpath_ewc,
                                                       lambda_ewc=config.lambda_ewc,
                                                       **training_args)
# Create the trainer:
trainer_args = dict(
    args=training_args,
    model=model,
    train_dataset=raw_dataset['train'],
    eval_dataset=raw_dataset['eval'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,  # use processor for saving the feature extractor
)


trainer = EWCFinetuningTrainer(**trainer_args)


print('TRAINING IN PROGRESS...')

trainer.train()

print('DONE TRAINING')
# trainer.save_model(f"{opt['output_dir']}")
