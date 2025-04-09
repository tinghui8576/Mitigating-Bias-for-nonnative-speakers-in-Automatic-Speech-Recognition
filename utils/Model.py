import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm

class Whisper(torch.nn.Module):
    def __init__(self, opt):
        super(Whisper, self).__init__()
        self.model_type = "openai/" +opt['model_type']
        self.processor = WhisperProcessor.from_pretrained(self.model_type)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_type)

        self.device = opt['device']

        self.model.to(self.device)

    def predict(self, dataloader):
        self.model.eval()
        # set the forced ids
        outputs_dic = {}

        for input_features, file_ids, splits in tqdm(dataloader, desc="Evaluating WER"):
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en",task="transcribe")
            with torch.cuda.amp.autocast():  # Mixed precision
                predicted_ids = self.model.generate(input_features,forced_decoder_ids = forced_decoder_ids)
            with torch.no_grad():
                outputs = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

            # Store the result
            for (file_id, split, output) in zip(file_ids, splits, outputs):
                # Initialize the list if it's the first time seeing the file_id
                if file_id not in outputs_dic:
                    outputs_dic[file_id] = []
                
                # Append the split and transcription tuple
                outputs_dic[file_id].append((split, output))

        for file_id, transcriptions in outputs_dic.items():
            # Sort the transcriptions by split_start and split_end to concatenate in the correct order
            sorted_transcriptions = sorted(transcriptions, key=lambda x: x[0])  # Sort by split_start, split_end
            # Concatenate all transcriptions in the correct order
            full_transcription = "".join([transcription for _, transcription in sorted_transcriptions])
            
            # Store the combined transcription for the file_id
            outputs_dic[file_id] = full_transcription
        output_list = [{'id': key, 'output': value} for key, value in outputs_dic.items()]
        # Create a DataFrame
        outputs_df = pd.DataFrame(output_list)

        return outputs_df