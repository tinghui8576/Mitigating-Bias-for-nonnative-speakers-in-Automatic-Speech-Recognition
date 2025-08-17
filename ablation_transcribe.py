import os
import torch
import argparse
from Model.Model import Whisper
from utils.language_map import language_family_dict, accent_map
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import Audio, load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from multiprocessing import Pool
from Model.utils import add_leace
from utils.useage import chunk_audio, transcribe_chunks, clean_transcript
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
    model_id = args.model_name
    if (args.is_public_repo == False) and (args.strategy != "leace"):
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
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        processor = WhisperProcessor.from_pretrained(model_id, language=args.language, task="transcribe")
        if args.strategy == "leace":
            model = add_leace(model, args.ckpt_dir)
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

    if (args_global.is_public_repo == False) and (args_global.strategy != "leace"):
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
    parser.add_argument("--model_name", type=str, required=False, default="openai/whisper-small")
    parser.add_argument("--strategy", type=str, required=False, default="lora")
    parser.add_argument("--ckpt_dir", type=str, required=False, default=".")
    parser.add_argument("--language", type=str, required=False, default="en")
    parser.add_argument("--eval_datasets", type=str, nargs='+', required=True, default=[])
    parser.add_argument("--device", type=int, required=False, default=0)
    parser.add_argument("--output_dir", type=str, required=False, default="predictions_dir")
    parser.add_argument("--outputs", type=str, required=False, default="test")
    args = parser.parse_args()

    main(args)
