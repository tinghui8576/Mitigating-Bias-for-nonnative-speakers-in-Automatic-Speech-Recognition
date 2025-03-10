import librosa
from torch.utils.data import Dataset
from transformers import WhisperProcessor

class WhisperDataset(Dataset):
    def __init__(self, df, opt = None):
        self.sample_rate=16000
        self.files = df['FilePath']
        self.file_ids = df['id']
        self.language = 'english'
        self.splits_start = df['split_start']
        self.splits_end = df['split_end']
        # opt.get('model_type',"openai/whisper-medium")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small",
                                                            language="en",
                                                            task="transcribe")
    
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        filename = self.files[index]
        file_id = self.file_ids[index]
        split_start = self.splits_start[index]
        split_end = self.splits_end[index]
        y, sr = librosa.load(filename, sr= self.sample_rate)
        # wav, sample_rate = torchaudio.load(filename, normalize=True)
     
        # if sample_rate != self.sample_rate:
        #     resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
        #     wav = resampler(wav)

        
        split_point = int(25 * self.sample_rate)
        total_length = len(y)
        split_end = min(split_point * split_end, total_length)  # Don't exceed the length
        audio_slice = y[ split_point * split_start : split_end]

        
        #input_features = torch.tensor(self.processor.feature_extractor(audio_slice.numpy(), sampling_rate=self.sample_rate).input_features[0])
        
        input_features = self.processor(audio_slice , sampling_rate=self.sample_rate, return_tensors="pt").input_features
        
        #input_ids = torch.tensor(self.processor.tokenizer(label).input_ids)
        #input_ids,
        return input_features, file_id, split_start
    
       