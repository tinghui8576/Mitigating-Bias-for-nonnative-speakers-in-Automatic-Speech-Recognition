import re
import pandas as pd
from difflib import SequenceMatcher
from torch.nn.utils.rnn import pad_sequence
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

def collate_fn(batch):
    input_features = [b[0] for b in batch]
    file_ids = [b[1] for b in batch]
    split = [b[2] for b in batch]
    # Pad the input features
    padded_input_features = pad_sequence(input_features, batch_first=True, padding_value=0.0)
    padded_input_features = padded_input_features.squeeze(1)
    # Ensure batch sizes are correct and display info for debugging
    # print(f"Batch size: {len(batch)}")
    # print(f"Padded input shape: {padded_input_features.shape}")
    return padded_input_features, file_ids, split

def remove_duplicates(input_file, output_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Remove duplicates across all columns
    df_cleaned = df.drop_duplicates()

    df_cleaned.to_csv(output_file, index=False)
    print(f"Duplicates removed and saved to {output_file}")

def extract_age(age_sex):
    try:
        # Using regex to find the age before the comma
        age_match = re.search(r"(\d+)", age_sex)  # \d+ matches one or more digits
        
        if age_match:
            return age_match.group(1)  # Return the first matched group (age)
        return None  # If no match is found, return None
    except Exception as e:
        print(f"Error while extracting age: {e}")
        return None

def clean_transcript(text, max_repeat=2):
    """
    Cleans transcript by:
    1. Removing repeated unigrams/bigrams/trigrams
    2. Removing common fillers and stutters
    """
    # Define fillers/stutters
    fillers = {"um", "uh", "er", "ah", "like", "hmm"}
    
    tokens = text.strip().split()
    cleaned_tokens = []
    i = 0
    while i < len(tokens):
        # Remove filler words
        token_lower = tokens[i].lower()
        if token_lower in fillers:
            i += 1
            continue

        # Remove repetitions
        repeat_skipped = False
        for n in (3, 2, 1):  # Check trigrams, bigrams, unigrams
            if i + 2 * n <= len(tokens):
                chunk = tokens[i:i+n]
                next_chunk = tokens[i+n:i+2*n]
                if chunk == next_chunk:
                    # Skip the repeated chunk
                    i += n
                    repeat_skipped = True
                    break
        if repeat_skipped:
            continue

        cleaned_tokens.append(tokens[i])
        i += 1
    return " ".join(cleaned_tokens)

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

def transcribe_chunks(chunks, sr, processor, model, device, forced_decoder_ids, accent_id=None):
    inputs = processor.feature_extractor(chunks, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    if accent_id is not None:
        generated_ids = model.generate(
            input_features,
            accent_id = accent_id,
            forced_decoder_ids=forced_decoder_ids
        )
    else:
        generated_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids
        )
    text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    merged = smart_merge(text)
    return merged
