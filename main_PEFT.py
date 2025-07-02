import os
import torch
import argparse
from utils.Model import WhisperPEFT
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import pipeline
from datasets import Audio, load_from_disk
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration

def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    else:
        return ref.strip() != ""


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


def get_text_column_names(column_names):
    if "text" in column_names:
        return "text"
    elif "sentence" in column_names:
        return "sentence"
    elif "normalized_text" in column_names:
        return "normalized_text"
    elif "transcript" in column_names:
        return "transcript"
    elif "transcription" in column_names:
        return "transcription"


whisper_norm = BasicTextNormalizer()
# def normalise(batch):
#     batch["norm_text"] = whisper_norm(get_text(batch))
#     return batch
def normalise(batch):
    text_col = get_text_column_names(batch.keys())
    return {
        "norm_text": [whisper_norm(t) for t in batch[text_col]]
    }

def chunk_audio(audio_array, sr, chunk_length_s=30, stride_s=6):
    chunk_samples = int(sr * chunk_length_s)
    stride_samples = int(sr * stride_s)
    chunks = []
    for start in range(0, len(audio_array), chunk_samples - stride_samples):
        end = min(start + chunk_samples, len(audio_array))
        chunk = audio_array[start:end]
        chunks.append(chunk)
        if end == len(audio_array):
            break
    return chunks

from difflib import SequenceMatcher

def smart_merge(predictions, min_overlap=3):
    if not predictions:
        return ""

    final_words = predictions[0].strip().split()

    for i in range(1, len(predictions)):
        curr_words = predictions[i].strip().split()

        # Use SequenceMatcher on word-level tokens
        matcher = SequenceMatcher(None, final_words, curr_words)
        match = matcher.find_longest_match(0, len(final_words), 0, len(curr_words))
        if match.size >= min_overlap:
            # Replace overlap: keep left before match, right after match
            left = final_words[:match.a]
            right = curr_words[match.b + match.size:]
            final_words = left + curr_words[match.b:match.b + match.size] + right
        else:
            # No meaningful overlap: just append
            final_words += curr_words

    return " ".join(final_words)


def transcribe_chunks(chunks, sr, processor, model, device, forced_decoder_ids):
    inputs = processor.feature_extractor(chunks, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    generated_ids = model.generate(
        input_features,
        forced_decoder_ids=forced_decoder_ids
    )
    text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    

    merged = smart_merge(text)
    return merged


def main(args):
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    opt = {
        "model_name": args.hf_model,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    if args.is_public_repo == False:
        # Load processor and custom Whisper model
        processor = WhisperProcessor.from_pretrained(args.ckpt_dir)
        model = WhisperPEFT(opt, processor)
        state_dict = torch.load(os.path.join(args.ckpt_dir, "best_model.pt"), map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)
        model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})

        # model.Biasadapter.load_state_dict({k.replace("module.Biasadapter.", ""): v for k, v in state_dict.items() if k.startswith("module.Biasadapter.")})
    else:
        model_id = args.hf_model
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        processor = WhisperProcessor.from_pretrained(model_id, language=args.language, task="transcribe")
    
    model.to(f"cuda:{args.device}" if args.device >= 0 else "cpu")

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language,task="transcribe")

    os.makedirs(args.output_dir, exist_ok=True)

    for dset in args.eval_datasets:
        print('\nInfering on the dataset : ', dset)
        dataset = load_from_disk(dset)
        text_column_name = get_text_column_names(dataset.column_names)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        # dataset = dataset.map(normalise, batched=True, batch_size=args.batch_size)
        # dataset = dataset.filter(is_target_text_in_range, input_columns=[text_column_name], num_proc=2)

        filenames, all_preds = [], []

        for item in tqdm(dataset, desc='Decode Progress'):
            audio = item["audio"]
            audio_array = audio["array"]
            sr = audio["sampling_rate"]
            chunks = chunk_audio(audio_array, sr, chunk_length_s=25, stride_s=5)
        
            chunk_preds = transcribe_chunks(chunks, sr, processor, model, f"cuda:{args.device}" if args.device >= 0 else "cpu", forced_decoder_ids)

            # Merge top-1 from each chunk for now
            # merged_prediction = " ".join([pred[0] for pred in chunk_preds])
            all_preds.append(chunk_preds)
            filenames.append(audio["path"].split("/")[-1].replace(".wav", ""))

        outputs_df = pd.DataFrame({"id": filenames, "prediction": all_preds})
        output_csv_path = os.path.join(args.output_dir, f"{args.outputs}.csv")
        outputs_df.to_csv(output_csv_path, index=False)


def data(dataset):
    for i, item in enumerate(dataset):
        yield {**item["audio"], "reference": get_text(item), "norm_reference": item["norm_text"]}

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
        "--hf_model",
        type=str,
        required=False,
        default="openai/whisper-small",
        help="Huggingface model name. Example: openai/whisper-tiny",
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
    # parser.add_argument(
    #     "--batch_size",
    #     type=int,
    #     required=False,
    #     default=16,
    #     help="Number of samples to go through each streamed batch.",
    # )
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

# python main2.py \           
# --is_public_repo True \
# --hf_model "openai/whisper-large"
# --language en \
# --eval_datasets CustomData/SSA \
# --device -1 \
# --batch_size 8 \
# --output_dir results \
# --outputs large

# python main2.py \           
# --is_public_repo False \
# --ckpt_dir "Model/test5" \
# --language en \
# --eval_datasets CustomData/SSA \
# --device -1 \
# --batch_size 8 \
# --output_dir results \
# --outputs test5_best