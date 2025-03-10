import re
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


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