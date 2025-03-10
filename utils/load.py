import os
import math
import utils
import librosa
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def walkthrough(audio_files, audio_dir, meta):
    audio_extensions = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
    audio_data = []
    for file in tqdm(audio_files, desc="Processing audio files", unit="file"):
        file_extension = Path(file).suffix.lower()
        if file_extension in audio_extensions:
            file_path = os.path.join(audio_dir,file)
            y, sr = librosa.load(file_path)
            duration = len(y) / sr
            if (duration < 0.1):
                print(f"{file} Less than 0 sec")
            else:
                if not meta[meta['id']==file[:-4]].empty:
                    if duration > 25:
                        times = math.ceil(duration/25)

                        for i in range(times):
                            audio_data.append([file[:-4],i, i+1])
                    else:
                        audio_data.append([file[:-4],0, 1])
    return pd.DataFrame(audio_data, columns=["id", "split_start", "split_end"])

def get_dataloader(opt):    

    
    path = os.path.join('Data', opt['dataset_name'])
    print(path)
    assert os.path.exists(path), f"Dataset file {path} does not exist."
    if opt['dataset_name'] == 'speech-accent-archive':
        utils.remove_duplicates(os.path.join(path,'bio.csv'),os.path.join(path,"bio.csv"))

        audio_dir = os.path.join(path, 'audio')
        audio_files = os.listdir(audio_dir)

        meta = pd.read_csv(os.path.join(path,'bio.csv'))
        assert not meta.empty, f"Dataset {opt['dataset_name']} is empty or could not be loaded."
        meta.rename(columns={'filename': 'id'}, inplace=True)
        meta['age'] = meta['age_sex'].apply(lambda x: utils.extract_age(x))
        meta = meta.dropna(subset=['age', 'sex', 'native_language', 'birth_place'])

        audio_df = walkthrough(audio_files, audio_dir, meta)

        meta = meta.drop(columns=['href'])

        meta['native_language'] = meta['native_language'].apply(lambda x: x.split('\n')[0])
        meta['sex'] = meta['sex'].replace('famale', 'female')
        df = pd.merge(meta, audio_df ,  on='id', how="inner")
        df['FilePath'] = df['id'].apply(lambda x: os.path.join(audio_dir,x+'.wav'))
        df['sentence'] = 'Please call Stella.  Ask her to bring these things with her from the store:  Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.'
        
    elif opt['dataset_name'] == 'artie-bias-corpus':
        audio_dir = path
        audio_files = os.listdir(audio_dir)

        meta_abc = pd.read_csv(os.path.join(path, 'artie-bias-corpus.tsv'), delimiter = '\t')
        
        assert not meta_abc.empty, f"Dataset {opt['dataset_name']} is empty or could not be loaded."
        meta_abc['id'] = meta_abc['path'].apply(lambda x: x[:-4])
        meta_abc = meta_abc.dropna(subset=['age', 'gender', 'accent', 'sentence'])

        audio_df = walkthrough(audio_files, audio_dir, meta_abc)

        df = pd.merge(meta_abc, audio_df ,  on='id', how="inner")

        # Drop rows where gender is 'other'
        df = df[df['gender'] != 'other']
        df = df[df['accent'] != 'other']
        df.reset_index(drop=True, inplace=True)
        df['FilePath'] = df['path'].apply(lambda x: os.path.join(audio_dir,x))
        
    print(f"Dataset {opt['dataset_name']} loaded successfully with {len(df)} records.")

    
    return df





