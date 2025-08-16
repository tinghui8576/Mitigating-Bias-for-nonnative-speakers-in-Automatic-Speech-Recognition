import os
import time
import wandb
import torch
import utils.constant as const
from functools import partial
from Model.Model import Whisper
from utils.parse import parse_args
from transformers import WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from DatasetLoad.dataset_loader import load_dataset_dict
from DatasetLoad.collator import DataCollatorSpeechSeq2SeqWithPadding
from Refer.trainer.ewc_finetuning import EWCFinetuningTrainer, EWCFinetuningTrainingArguments
from Refer.utils.ewc_finetune_config import EWCFinetuneConfig
from Refer.trainer.lwf_finetuning import LwFTrainer, LwFTrainingArguments


# #######################     ARGUMENT PARSING        #########################
opt = vars(parse_args())

# Check if a GPU is available
opt['device'] = torch.device('cuda' if opt['use_cuda'] and torch.cuda.is_available() else 'cpu')

if opt['wandb'] == True:
    wandb.login()
    
    run_name = f"whisper_finetune_{int(time.time())}"
    wandb.init(project="whisper-finetune", config=opt, name=run_name)

const.num_proc  = opt['num_proc']
const.random_state = opt['random_state']

# ############################        DATASET LOADING AND PREP        ##########################
processor = WhisperProcessor.from_pretrained(opt["model_name"], language=opt["language"], task="transcribe")

dataset_dic = load_dataset_dict(processor)

# ###############################     DATA COLLATOR AND METRIC DEFINITION     ########################

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

print('DATASET PREPARATION COMPLETED')
from Evaluate.compute_metric import compute_metrics_fct
compute_metrics = partial(compute_metrics_fct,
                        processor=processor)


def custom_save_model(output_dir, _internal_call=False):
    os.makedirs(output_dir, exist_ok=True)
    # Save full wrapper model (including adapter layer) and processor
    torch.save(model.state_dict(), os.path.join(output_dir, "full_model.pth"))
    model.processor.save_pretrained(output_dir)

# #############################       MODEL LOADING       #####################################
model = Whisper(opt)
model.train()

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"✅ Trainable: {name} \t shape={tuple(param.shape)}")
    # else:
        # print(f"❌ Frozen:    {name} \t shape={tuple(param.shape)}")


# ###############################     TRAINING ARGS AND TRAINING      ############################

training_args_dict = dict(
        output_dir=opt['output_dir'],
        per_device_train_batch_size=opt['train_batchsize'],
        gradient_accumulation_steps=2,
        learning_rate=opt['learning_rate'],
        warmup_steps=opt['warmup'],
        gradient_checkpointing=const.gradient_checkpointing,
        # fp16=True,
        bf16=True,
        per_device_eval_batch_size=opt['eval_batchsize'],
        predict_with_generate=True,
        generation_max_length=300,
        num_train_epochs = 1,

        # evaluation_strategy="steps",
        eval_strategy="steps",
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
        
        # optim="adamw_bnb_8bit",
        optim="adamw_torch",
        resume_from_checkpoint=opt['resume_from_ckpt'],

        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],
    )

if opt['strategy'].lower() == 'ewc':
    config = EWCFinetuneConfig.from_yaml("Refer/config/test.yaml")
    training_args = EWCFinetuningTrainingArguments(dirpath_ewc=config.dirpath_ewc,
                                                        lambda_ewc=config.lambda_ewc,
                                                        **training_args_dict)
elif opt['strategy'].lower() == 'lwf':
    training_args = LwFTrainingArguments(old_model_name=opt['model_name'],
                                    temp_lwf=2.0,
                                    alpha_lwf=0.5,
                                    **training_args_dict)
else:
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

# Create the trainer:
trainer_args = dict(
    args=training_args,
    model=model,
    train_dataset=dataset_dic['train'],
    eval_dataset=dataset_dic['eval'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,  # use processor for saving the feature extractor
)

if opt['strategy'].lower() == 'ewc':
    trainer = EWCFinetuningTrainer(**trainer_args)
elif opt['strategy'].lower() == 'lwf':
    trainer = LwFTrainer(**trainer_args)
else:
    trainer = Seq2SeqTrainer(**trainer_args)

trainer.save_model = custom_save_model

print('TRAINING IN PROGRESS...')

trainer.train()

print('DONE TRAINING')
trainer.save_model(f"{opt['output_dir']}")
