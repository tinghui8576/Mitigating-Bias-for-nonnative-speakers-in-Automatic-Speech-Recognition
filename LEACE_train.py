import torch
import os
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperModel
from concept_erasure import LeaceEraser
import argparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from DatasetLoad.dataset_loader import load_dataset_dict
#######################     ARGUMENT PARSING        #########################
import transformers
print(transformers.__version__)
print(transformers.__file__)
def parse_args():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Train a speech classification model with wandb logging.")
    # Define arguments
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
        '--batchsize', 
        type=int, 
        required=False, 
        default=48, 
        help='Batch size during the training phase.'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=False, 
        default='output_model_dir', 
        help='Output directory for the checkpoints generated.'
    )
    parser.add_argument('--use_cuda', action='store_true', help="Flag to use GPU if available.")
      
    return parser.parse_args()

opt = vars(parse_args())
# Check if a GPU is available
device = torch.device('cuda' if opt['use_cuda'] and torch.cuda.is_available() else 'cpu')

############################        DATASET LOADING AND PREP        ##########################
processor = WhisperProcessor.from_pretrained(opt["model_name"], language=opt["language"], task="transcribe")

raw_dataset = load_dataset_dict(processor)
#############################       MODEL LOADING       #####################################
model = WhisperModel.from_pretrained("openai/whisper-small").to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model.eval()

output_dir = opt['output_dir']
os.makedirs(output_dir, exist_ok=True)
BATCH_SIZE = opt['batchsize']

def process_dataset(dataset, layer_idx):
    X_tokens, y_labels = [], []

    for example in tqdm(dataset):
        audio_array = example["audio"]["array"]
        accent = example["accent_id"]

        # Process single audio
        inputs = processor(audio_array , sampling_rate=16000, return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["decoder_input_ids"] = torch.tensor([[50258]]).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        reps = outputs.encoder_hidden_states[layer_idx].squeeze(0)  # shape: (T, H)

        for token in reps.cpu().numpy():
            X_tokens.append(token)
            y_labels.append(accent)

    X = torch.tensor(X_tokens, dtype=torch.float32)
    y = torch.tensor(y_labels, dtype=torch.float32)
    return X, y

for layer_idx in range(12):
    print(f"\n[Layer {layer_idx}]")

    # --- Train phase ---
    X_train, y_train = process_dataset(raw_dataset["train"], layer_idx)
    eraser = LeaceEraser.fit(X_train, y_train)
    torch.save(eraser, os.path.join(output_dir, f"leace_eraser_layer{layer_idx}.pt"))

    # --- Eval phase ---
    X_test, y_test = process_dataset(raw_dataset["eval"], layer_idx)
    X_test = X_test.to(torch.float32) 

    # Accuracy before erasure
    clf = SGDClassifier(loss="hinge", max_iter=3000, tol=1e-3, random_state=42)
    clf.fit(X_test, y_test)
    acc_before = accuracy_score(y_test, clf.predict(X_test))

    # Accuracy after erasure
    X_erased = eraser(X_test)
    clf = SGDClassifier(loss="hinge", max_iter=3000, tol=1e-3, random_state=42)
    clf.fit(X_erased, y_test)
    acc_after = accuracy_score(y_test, clf.predict(X_erased))

    print(f"[✓] Saved LEACE for Layer {layer_idx}")
    print(f"Accuracy Before: {acc_before:.4f} | After: {acc_after:.4f}")

print('DONE TRAINING')