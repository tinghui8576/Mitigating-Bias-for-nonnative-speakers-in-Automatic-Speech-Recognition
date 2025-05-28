import os
import math
import torch
import utils
import librosa
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, load_from_disk, concatenate_datasets, Audio
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperProcessor
from datasets import disable_progress_bar
disable_progress_bar()
accent_map = {
    "Afro-Asiatic": 0,
    "Austro-Asiatic": 1,
    "Indo-European": 2, 
    "Japonic": 3,
    "Koreanic":4,
    "Sino-Tibetan":5,
    "Turkic": 6,
    "NorthAmerica": 7,
    "Europe": 8,
    "Australian": 9,
    "Africa": 10,
    "Caribbean": 11,
    "Asia": 12,
    "Austronesian": 13
    }


language_family = {
            "NorthAmerica": ['us', 'canada', 'mainstream us english'],
            "Europe": ['uk', 'england', 'wales', 'scotland', 'ireland', 'southern british english', 'english', 'scottish english', 'irish english'],
            "Australian": ['australia', 'newzealand'],
            "Africa": ['african', 'south african english', 'southatlandtic', 'nigerian english', "kenyan english", "ghanain english"],
            "Caribbean": ['jamaican', 'trinidadian', 'barbadian', 'bermuda', 'jamaican english' ],
            "Asia": ['singapore', 'hongkong', 'philippines', 'malaysia', 'indian', 'indian english', 'sinhalese'],

            "Austronesian": ['indonesian', 'malay', 'carolinian', 'cebuano', 'chamorro','fijian', 'filipino', 'gaddang', 'hiligaynon', 'javanese',
                        'lamaholot', 'malagasy', 'masbatenyo', 'mortlockese', 'pohnpeian', 'rotuman', 'satawalese', 'tagalog', 'yapese',
                        'lamotrekese', 'sundanese', 'bahasa' ],
            "Afro-Asiatic": ['amharic', 'arabic', 'chaldean', 'gedeo', 'hadiyya', 'hausa', 'hebrew', 'kabyle', 'kambaata', 'maltese', 'oromo',
                'sarua', 'somali', 'tigre', 'tigrigna', 'amazigh', 'maltese'],
            "Austro-Asiatic": ['khmer', 'vietnamese'],
            "Dravidian": ['kannada', 'malayalam', 'tamil', 'telugu'],
            "Eskimo-Aleut": ['yupik'],
            "Hmong-Mien": ['hmong'],
            "Isolate": ['basque'],
            "Indo-European": ['afrikaans', 'albanian', 'armenian', 'bengali', 'bavarian', 'bosnian','bulgarian', 'catalan', 'chittagonian', 
                                'croatian', 'czech', 'danish', 'dari', 'dutch','english', 'spanish', 'swedish', 'faroese', 'french', 'frisian',
                                'german', 'greek', 'gujarati', 'hindi', 'hindko', 'icelandic', 'italian', 'konkani', 'kurdish',
                                'latvian', 'lithuanian', 'macedonian', 'marathi', 'nepali', 'norwegian', 'serbian', 'slovak', 'oriya', 'ossetic',
                                'punjabi', 'pashto', 'polish', 'portuguese', 'russian', 'romanian', 'sardinian', 'sindhi', 'sinhala', 'sicilian',
                                'swiss', 'sylheti', 'ukrainian', 'urdu', 'yiddish', 'belarusan', 'charapa-spanish', 'farsi', 'vlaams', 'luxembourgeois',
                                'slovenian', 'tajiki', 'montenegrin', "portoguese", "spanish (mexican)"],
                                
            "Japonic": ['japanese'],
            "Kartvelian": ['georgian' ],
            "Koreanic": ['korean'],
            "Kra-Dai": ['lao', 'shan', 'thai'],
            "Maipurean":  ['garifuna'],
            "Mongolic": ['mongolian'],
            "Misumalpan": ['miskito'],
            "Niger-Congo": ['akan', 'baga', 'agni', 'bafang', 'zulu', 'bamun', 'chichewa', 'jola', 'ebira', 'edo', 'ewe', 'fang', 'fulfulde', 'ga',
                            'ganda', 'ibibio', 'ife', 'igbo', 'kalanga', 'kamba', 'kikongo', 'lingala', 'luba-kasai', 'rwanda', 'rundi', 'mankanya',
                            'mandinka', 'mende', 'moba', 'moore', 'ndebele', 'ngemba', 'pulaar', 'shona', 'susu', 'tumbuka', 'voro', 'wolof', 'yoruba',
                            'ashanti', 'fanti', 'twi', 'balanta', 'bambara', 'bayangi', 'gusii', 'kikuyu', 'kiswahili', 'kru', 'tswana', 'temne'],
            "Nilo-Saharan": ['bari','dinka', 'kanuri', 'nandi', 'nuer', 'shilluk', 'wali', 'luo', 'mandingo', 'serer', 'sesotho', 'xasonga'],
            "Sino-Tibetan": ['bai', 'burmese', 'cantonese', 'mizo', 'hakka', 'mandarin', 'min', 'hainanese' ,'naxi', 'taiwanese','wu', 'xiang', 'pahari',
                            'tibetan', 'gan', 'teochew', 'newari', 'taishan', 'chinese'],
            "South-Central Papuan": ['nama'],
            "Turkic": ['azerbaijani', 'kazakh', 'kyrgyz', 'tatar', 'turkish', 'turkmen', 'uyghur', 'uzbek', 'yakut'],
            "Trans-New Guinea": ['fataluku'],
            "Uralic": ['finnish', 'estonian', 'hungarian'],
            "Sign": ['home'],
            "Pidgin": ['liberian'],
            "Creole": ['krio', 'haitian', 'mauritian', 'papiamentu'],
            "Quechuan": ['quechua'],
            "Other": ['synthesized', 'northern', 'tok' ],
}
language_family_dict = {lang: family for family, langs in language_family.items() for lang in langs}

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
        df['sentence'] = 'Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.'
        
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


do_lower_case = True
do_remove_punctuation = True
normalizer = BasicTextNormalizer()

# max_label_length = model.config.max_length
max_label_length = 512
min_input_length = 0.0
max_input_length = 30.0
def is_in_length_range(length, labels):
    return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length

def prepare_dataset(batch, processor):
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    if not isinstance(audio, dict) or "array" not in audio or audio["array"] is None:
        print("⚠️ Problem with audio input:", audio)
        return {}

    if torch.isnan(torch.tensor(audio["array"])).any():
        print("⚠️ Found NaNs in audio array")

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    batch["accent_id"]= accent_map[language_family_dict [batch["accents"].lower()]]
    
    # optional pre-processing steps
    transcription = batch["sentence"]
    
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    if transcription is None or len(transcription.strip()) == 0:
        print("⚠️ Empty transcription:", transcription)

    try:
        batch["labels"] = processor.tokenizer(transcription).input_ids
        invalid_ids = [tid for tid in batch["labels"] if tid < 0 or tid >= processor.tokenizer.vocab_size]
        if invalid_ids:
            decoded = processor.tokenizer.decode(batch["labels"])
            print(f"\n⚠️ Transcription: {transcription}")
            print(f"⚠️ Tokenized to: {batch["labels"]}")
            print(f"⚠️ Decoded back: {decoded}")
            print(f"⚠️ Invalid token IDs: {invalid_ids}\n")
    except Exception as e:
        print("⚠️ Error tokenizing:", transcription, "\n", e)
        batch["labels"] = []
        
    return batch


# Remove rows with the <> pattern or "IGNORE_TIME_SEGMENT_IN_SCORING" in the sentence
def filter_invalid_sentences(example):
    import re
    # Regex pattern to find any text enclosed in angle brackets
    if re.search(r"<[^>]+>", example.get("sentence", "")):
        return False  # Exclude the sentence
    if example.get("sentence", "").startswith("IGNORE_TIME_SEGMENT_IN_SCORING"):
        return False
    return True  # Include the sentence
def load_all_datasets(opt, processor, split):
    
    if split == 'train':
        datalist = [
            {"Name": "edinburghcstr/edacc", "Type": "load", "Split": "validation", "Col_trans": "text", "Col_Accent": "l1"},
            # {"Name": "mispeech/speechocean762", "Type": "load", "Split": "train", "Column": "text"},
            # {"Name": "tobiolatunji/afrispeech-200", "Parts": "swahili", "Type": "load", "Split": "train", "Column": "transcript"},
            # {"Name": "tobiolatunji/afrispeech-200", "Parts": "swahili", "Type": "load", "Split": "validation", "Column": "transcript"},
            # {"Name": "tobiolatunji/afrispeech-200", "Parts": "igbo", "Type": "load", "Split": "train", "Column": "transcript"},
            # {"Name": "tobiolatunji/afrispeech-200", "Parts": "igbo", "Type": "load", "Split": "validation", "Column": "transcript"},
            # {"Name": "tobiolatunji/afrispeech-200", "Parts": "yoruba", "Type": "load", "Split": "train", "Column": "transcript"},
            # {"Name": "tobiolatunji/afrispeech-200", "Parts": "yoruba", "Type": "load", "Split": "validation", "Column": "transcript"},
            # {"Name": "l2arctic/shards", "Type": "disk", "Split": "None", "Column": "sentence"},
            # {"Name": "CustomData/commonvoice/train", "Type": "disk", "Split": "None", "Col_trans": "sentence", "Col_Accent": "accents"},
        ]
    if split == 'eval':
        datalist = [
            {"Name": "CustomData/commonvoice/test", "Type": "disk", "Split": "None", "Col_trans": "sentence", "Col_Accent": "accents" },
            # {"Name": "mispeech/speechocean762", "Type": "load", "Split": "test", "Column": "text"},
            {"Name": "edinburghcstr/edacc", "Type": "load", "Split": "test", "Col_trans": "text", "Col_Accent": "l1"},
        ]

    processed_datasets = []

    for ds in datalist:
        print(f"Loading: {ds['Name']} ({ds['Split']})")
        # Location = shared_disk + "/dataset"
        Location = "/Users/tinghui/.cache/huggingface/datasets"

        # Load dataset
        if ds["Type"] == "load":
            if ds.get("Parts"):
                dataset = load_dataset(ds["Name"], ds["Parts"], split=ds["Split"], cache_dir=Location)
            else:
                dataset = load_dataset(ds["Name"], split=ds["Split"], cache_dir=Location)
        else:
            # dataset = load_from_disk(os.path.join(Location, ds["Name"]))
            dataset = load_from_disk(ds["Name"])
        print(dataset)

        # Normalize and prep
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        if ds["Col_trans"] != "sentence":
            dataset = dataset.rename_column(ds["Col_trans"], "sentence")
        if ds["Col_Accent"] != "accents":
            dataset = dataset.rename_column(ds["Col_Accent"], "accents")
        if ds["Name"]=="edinburghcstr/edacc":
            dataset = dataset.filter(filter_invalid_sentences,desc=None)
        # dataset = dataset.filter(lambda x: "IGNORE_TIME_SEGMENT_IN_SCORING" not in x.get("sentence", ""))
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["audio", "sentence", "accents"]])
        dataset = dataset.select([10])
        # Apply mapping + filtering before combining
        dataset = dataset.map(lambda batch: prepare_dataset(batch, processor), num_proc=opt['num_proc'],desc=None)
        dataset = dataset.filter(
            is_in_length_range,
            input_columns=["input_length", "labels"],
            num_proc=opt['num_proc'],
            desc=None
        ) 
        processed_datasets.append(dataset)

    # Now combine
    return concatenate_datasets(processed_datasets).shuffle(seed=22)


