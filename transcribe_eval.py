import os
import torch
import argparse
from Model.Model import Whisper
from utils.language_map import language_family_dict, accent_map
import pandas as pd
from tqdm import tqdm
from datasets import Audio, load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from utils.useage import chunk_audio, transcribe_chunks, clean_transcript
from utils.eval import metrics
from utils.constant import sentence
import re

def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', 'normalized_text' or 'transcript'. Got sample of "
            ".join{sample.keys()}. Ensure a text column name is present in the dataset."
        )


def main(args):
    args.device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    if args.is_public_repo == False:
        # Load processor and custom Whisper model
        processor = WhisperProcessor.from_pretrained(args.ckpt_dir)
        model = Whisper(vars(args))
        if args.strategy == "lora":
            state_dict = torch.load(os.path.join(args.ckpt_dir, "best_model.pt"), map_location=args.device)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
            model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
        elif args.strategy == "lwf":
            model.load_state_dict(torch.load(f"{args.ckpt_dir}/full_model.pth", map_location=args.device))
        model.eval()
    else:
        model_id = args.model_name
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        processor = WhisperProcessor.from_pretrained(model_id, language=args.language, task="transcribe")
    
    model.to(args.device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language,task="transcribe")

    os.makedirs(args.output_dir, exist_ok=True)

    for dset in args.eval_datasets:
        print('\nInfering on the dataset : ', dset)
        dataset = load_from_disk(dset)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        filenames, all_preds, all_performances = [], [], []

        for item in tqdm(dataset, desc='Decode Progress'):
            audio = item["audio"]
            audio_array = audio["array"]
            accent = item["accents"]
            sr = audio["sampling_rate"]
            chunks = chunk_audio(audio_array, sr, chunk_length_s=25, stride_s=5)
            if args.is_public_repo == False:
                chunk_preds = transcribe_chunks(chunks, sr, processor, model, args.device, forced_decoder_ids, accent_id=torch.tensor([accent_map[language_family_dict[accent]]]))
            else:
                chunk_preds = transcribe_chunks(chunks, sr, processor, model,args.device, forced_decoder_ids)
            chunk_preds = clean_transcript(chunk_preds)
            performance = metrics(chunk_preds, sentence)
            all_preds.append(chunk_preds)
            all_performances.append(performance)
            filenames.append(audio["path"].split("/")[-1].replace(".wav", ""))


        outputs_df = pd.DataFrame({"id": filenames, "prediction": all_preds, "performance": all_performances})
        output_csv_path = os.path.join(args.output_dir, f"{args.outputs}.csv")
        outputs_df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--is_public_repo",
        required=False,
        default=True, 
        type=lambda x: (str(x).lower() == 'true'),
        help="If the model is available for download on huggingface.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="openai/whisper-small",
        help="Huggingface model name. Example: openai/whisper-tiny",
    )
    parser.add_argument(
        '--strategy', 
        type=str, 
        required=False, 
        # choices=['lora', 'lwf'], 
        help='Finetune Strategy for Whisper model.'
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=False,
        default=".",
        help="Folder with the pytorch_model.bin file",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=False,
        default="en",
        help="Two letter language code for the transcription language, e.g. use 'hi' for Hindi. This helps initialize the tokenizer.",
    )
    parser.add_argument(
        "--eval_datasets", 
        type=str, 
        nargs='+', 
        required=True, 
        default=[], 
        help="List of datasets to evaluate the model on."
    )
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="The device to run the pipeline on. -1 for CPU, 0 for the first GPU (default) and so on.",
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=False, 
        default="predictions_dir", 
        help="Output directory for the predictions and hypotheses generated.")
    parser.add_argument(
        "--outputs", 
        type=str, 
        required=False, 
        default="" \
        "test", 
        help="Output directory for the predictions and hypotheses generated.")

    args = parser.parse_args()
    main(args)
