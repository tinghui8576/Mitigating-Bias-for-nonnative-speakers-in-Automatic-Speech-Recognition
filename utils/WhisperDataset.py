import torch
import librosa
from torch.utils.data import Dataset
from transformers import WhisperProcessor

accent_map = {
    "Afro-Asiatic": 0,
    "Austro-Asiatic": 1,
    "Indo-European": 2, 
    "Japonic": 3,
    "Koreanic":4,
    "Sino-Tibetan":5,
    "Turkic": 6
    }
language_family = {
            "Austronesian": ['indonesian', 'malay', 'carolinian', 'cebuano', 'chamorro','fijian', 'filipino', 'gaddang', 'hiligaynon', 'javanese',
                'lamaholot', 'malagasy', 'masbatenyo', 'mortlockese', 'pohnpeian', 'rotuman', 'satawalese', 'tagalog', 'yapese',
                'lamotrekese', 'sundanese'],
            "Afro-Asiatic": ['amharic', 'arabic', 'chaldean', 'gedeo', 'hadiyya', 'hausa', 'hebrew', 'kabyle', 'kambaata', 'maltese', 'oromo',
                'sarua', 'somali', 'tigre', 'tigrigna', 'amazigh'],
            "Austro-Asiatic": ['khmer', 'vietnamese'],
            "Dravidian": ['kannada', 'malayalam', 'tamil', 'telugu'],
            "Eskimo-Aleut": ['yupik'],
            "Hmong-Mien": ['hmong'],
            "Isolate": ['basque'],
            "Indo-European": ['afrikaans', 'albanian', 'armenian', 'bengali', 'bavarian', 'bosnian','bulgarian', 'catalan', 'chittagonian', 
                                'croatian', 'czech', 'danish', 'dari', 'dutch','english', 'spanish', 'swedish', 'faroese', 'french', 'frisian',
                                'german', 'greek', 'gujarati', 'hindi', 'hindko', 'icelandic', 'irish', 'italian', 'konkani', 'kurdish',
                                'latvian', 'lithuanian', 'macedonian', 'marathi', 'nepali', 'norwegian', 'serbian', 'slovak', 'oriya', 'ossetic',
                                'punjabi', 'pashto', 'polish', 'portuguese', 'russian', 'romanian', 'sardinian', 'sindhi', 'sinhala', 'sicilian',
                                'swiss', 'sylheti', 'ukrainian', 'urdu', 'yiddish', 'belarusan', 'charapa-spanish', 'farsi', 'vlaams', 'luxembourgeois',
                                'slovenian', 'tajiki'],
                                
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
                            'tibetan', 'gan', 'teochew', 'newari', 'taishan'],
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
class WhisperDataset(Dataset):
    def __init__(self, df, opt = None):
        self.sample_rate=16000
        self.files = df['FilePath']
        self.file_ids = df['id']
        self.scripts = df['transcript']
        self.accents = df['accent']
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        # opt.get('model_type',"openai/whisper-medium")
    
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        filename = self.files[index]

        audio, sr = librosa.load(filename, sr= self.sample_rate)
        prompt_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")

        input_features = self.processor.feature_extractor(audio, sampling_rate=self.sample_rate, return_tensors="pt").input_features
        
        accent_id = accent_map[language_family_dict [self.accents[index]]]
        transcript = self.scripts[index]

        # tokens = self.processor.tokenizer(transcript).input_ids
        # normalized_transcript = self.processor.tokenizer._normalize(transcript)
        tokens = self.processor.tokenizer(transcript).input_ids
        
        # tokens = torch.tensor([tokens], dtype=torch.long)
        #input_features = torch.tensor(self.processor.feature_extractor(audio_slice.numpy(), sampling_rate=self.sample_rate).input_features[0])
        # input_features = self.processor.feature_extractor(audio , sampling_rate=self.sample_rate, return_tensors="pt").input_features

        # tokens = self.processor.tokenizer(self.scripts[index]).input_ids
        
        return {
            'input_features': input_features.squeeze(0),
            'tokens':  torch.tensor(tokens, dtype=torch.long),
            'accent_id': accent_id
        }
    
       