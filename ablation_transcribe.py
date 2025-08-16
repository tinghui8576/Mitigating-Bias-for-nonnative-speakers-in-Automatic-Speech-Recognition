import os
import torch
import argparse
from Model.Model import Whisper
from utils.language_map import language_family_dict, accent_map
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import Audio, load_from_disk
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from difflib import SequenceMatcher
from functools import partial
from multiprocessing import Pool

# ===============================
# Globals (per process)
# ===============================
model = None
processor = None
forced_decoder_ids = None
args_global = None
layer_idx_global = None

# ===============================
# Utility Functions
# ===============================

def clean_transcript(text, max_repeat=2):
    fillers = {"um", "uh", "er", "ah", "like", "hmm"}
    tokens = text.strip().split()
    cleaned_tokens = []
    i = 0
    while i < len(tokens):
        token_lower = tokens[i].lower()
        if token_lower in fillers:
            i += 1
            continue
        repeat_skipped = False
        for n in (3,2,1):
            if i + 2*n <= len(tokens):
                chunk = tokens[i:i+n]
                next_chunk = tokens[i+n:i+2*n]
                if chunk == next_chunk:
                    i += n
                    repeat_skipped = True
                    break
        if repeat_skipped:
            continue
        cleaned_tokens.append(tokens[i])
        i += 1
    return " ".join(cleaned_tokens)

def chunk_audio(audio_array, sr, chunk_length_s=30, stride_s=6):
    chunk_samples = int(sr * chunk_length_s)
    stride_samples = int(sr * stride_s)
    chunks = []
    for start in range(0, len(audio_array), chunk_samples - stride_samples):
        end = min(start + chunk_samples, len(audio_array))
        chunks.append(audio_array[start:end])
        if end == len(audio_array):
            break
    return chunks

def smart_merge(predictions, min_overlap=3):
    if not predictions:
        return ""
    final_words = predictions[0].strip().split()
    for i in range(1, len(predictions)):
        curr_words = predictions[i].strip().split()
        matcher = SequenceMatcher(None, final_words, curr_words)
        match = matcher.find_longest_match(0, len(final_words), 0, len(curr_words))
        if match.size >= min_overlap:
            left = final_words[:match.a]
            right = curr_words[match.b+match.size:]
            final_words = left + curr_words[match.b:match.b+match.size] + right
        else:
            final_words += curr_words
    return " ".join(final_words)

def transcribe_chunks(chunks, sr, processor, model, device, forced_decoder_ids, accent_id=None):
    inputs = processor.feature_extractor(chunks, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    if accent_id is not None:
        generated_ids = model.generate(
            input_features,
            accent_id=accent_id,
        )
    else:
        generated_ids = model.generate(input_features)
    text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return smart_merge(text)

def skip_layer_output(module, input, output):
    if isinstance(output, tuple):
        return (input[0],) + output[1:]
    else:
        return input[0]

# ===============================
# Initializer
# ===============================
def init_worker(args, layer_idx):
    global model, processor, forced_decoder_ids, args_global, layer_idx_global

    args_global = args
    layer_idx_global = layer_idx

    if args.is_public_repo == False:
        processor = WhisperProcessor.from_pretrained(args.ckpt_dir)
        model = Whisper({
            "model_name": args.hf_model,
            "strategy": args.strategy,
            "language": args.language,
            "device": "cpu"
        })
        model.load_state_dict(
            torch.load(
                f"/work3/s232855/Models/{args.ckpt_dir}/full_model.pth",
                map_location="cpu"
            )
        )
        model.eval()
    else:
        model = WhisperForConditionalGeneration.from_pretrained(args.hf_model)
        processor = WhisperProcessor.from_pretrained(
            args.hf_model, language=args.language, task="transcribe"
        )
    model.to("cpu")

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task="transcribe"
    )

    # Register skip hook ONCE per process
    if args.is_public_repo == False:
        model._encoder_layers[layer_idx].register_forward_hook(skip_layer_output)
    else:
        model.model.encoder.layers[layer_idx].register_forward_hook(skip_layer_output)

# ===============================
# Worker Function
# ===============================
def process_single_file(item):
    global model, processor, forced_decoder_ids, args_global, layer_idx_global

    audio = item["audio"]
    audio_array = audio["array"]
    accent = item["accents"]
    sr = audio["sampling_rate"]
    chunks = chunk_audio(audio_array, sr, chunk_length_s=25, stride_s=5)

    if args_global.is_public_repo == False:
        chunk_preds = transcribe_chunks(
            chunks, sr, processor, model, "cpu", forced_decoder_ids,
            accent_id=torch.tensor([accent_map[language_family_dict[accent]]])
        )
    else:
        chunk_preds = transcribe_chunks(
            chunks, sr, processor, model, "cpu", None
        )

    chunk_preds = clean_transcript(chunk_preds)
    filename = audio["path"].split("/")[-1].replace(".wav", "")

    return {
        "id": filename,
        "prediction": chunk_preds,
        "skipped_layer": layer_idx_global
    }

# ===============================
# Main Logic
# ===============================
def main(args):
    dataset_path = args.eval_datasets[0]
    dataset = load_from_disk(dataset_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    os.makedirs(args.output_dir, exist_ok=True)

    encoder_layer_indices = list(range(12))  # For example, layers 0–11

    for layer_idx in encoder_layer_indices:
        print(f"\n[INFO] Skipping Encoder Layer {layer_idx}")

        # Pool with initializer
        with Pool(
            processes=4,
            initializer=init_worker,
            initargs=(args, layer_idx)
        ) as pool:
            results = list(tqdm(pool.imap(process_single_file, dataset), total=len(dataset)))

        df_layer = pd.DataFrame(results)

        # Save per layer
        df_layer.to_csv(os.path.join(
            args.output_dir,
            f"{args.outputs}_layer{layer_idx}.csv"
        ), index=False)


# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_public_repo", required=False, default=True,
        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--hf_model", type=str, required=False, default="openai/whisper-small")
    parser.add_argument("--strategy", type=str, required=False, default="lora")
    parser.add_argument("--ckpt_dir", type=str, required=False, default=".")
    parser.add_argument("--language", type=str, required=False, default="en")
    parser.add_argument("--eval_datasets", type=str, nargs='+', required=True, default=[])
    parser.add_argument("--device", type=int, required=False, default=0)
    parser.add_argument("--output_dir", type=str, required=False, default="predictions_dir")
    parser.add_argument("--outputs", type=str, required=False, default="test")
    args = parser.parse_args()

    main(args)
