import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torchmetrics.text import WordInfoLost, WordErrorRate,CharErrorRate, MatchErrorRate
from hyperpyyaml import load_hyperpyyaml
from whisper.normalizers import EnglishTextNormalizer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import pandas as pd
from tqdm import tqdm

WER = WordErrorRate()
WIL = WordInfoLost()
CER = CharErrorRate()
MER = MatchErrorRate()
ember_hparams = load_hyperpyyaml("""
wer_stats: !new:speechbrain.utils.metric_stats.ErrorRateStats

ember_embeddings: !apply:speechbrain.lobes.models.flair.embeddings.FlairEmbeddings.from_hf
    embeddings_class: !name:flair.embeddings.FastTextEmbeddings
    source: facebook/fasttext-en-vectors
    save_path: ./pretrained_models/

ember_metric: !new:speechbrain.utils.metric_stats.EmbeddingErrorRateSimilarity
    embedding_function: !name:speechbrain.lobes.models.flair.embeddings.FlairEmbeddings.embed_word
        - !ref <ember_embeddings>
    low_similarity_weight: 1.0
    high_similarity_weight: 0.1
    threshold: 0.4

weighted_wer_stats: !new:speechbrain.utils.metric_stats.WeightedErrorRateStats
    base_stats: !ref <wer_stats>
    cost_function: !ref <ember_metric>
    weight_name: ember
""")

    
class SemDistCalculator:
    def __init__(self, model_name="roberta-base", alpha = 1000, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.alpha = alpha

    def embed_sentence(self, sentence):
        sentence = [" ".join(sent) for sent in sentence]
        
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs).last_hidden_state  
            mask = inputs["attention_mask"] #.expand(outputs.size())  
            masked_outputs = outputs * mask.unsqueeze(-1) 
            nonmasked_counts = torch.sum(mask, dim=-1)
            mean_pooled = torch.sum(masked_outputs, dim=-2) / nonmasked_counts.unsqueeze(-1)
    
        return mean_pooled.squeeze(0)

    def safe_semdist(self, sim: float, eps: float = 1e-4):
        semdist = (1 - sim) * self.alpha
        return 0.0 if semdist < eps else semdist.item()
    
    def semdist(self, pred: str, ref: str):
        emb1 = self.embed_sentence(pred)
        emb2 = self.embed_sentence(ref)
        sim = F.cosine_similarity(emb1, emb2, dim=0)
        sim = self.safe_semdist(sim)

        return sim  # cosine distance
semdist = SemDistCalculator("xlm-roberta-base", alpha = 100000)



eng_normalizer = EnglishTextNormalizer()
normalizer = BasicTextNormalizer()


def normal(text):
    if text.startswith(" "):
        text = text[1:]
    
    text = eng_normalizer(text)
    # text = normalizer(text)
    return text


df = pd.read_csv('Data/speech-accent-archive/bio.csv')

df.rename(columns={'filename': 'id'}, inplace=True)
df['sentence'] = 'Please call Stella.  Ask her to bring these things with her from the store:  Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.'
df['accent'] = df['native_language'].apply(lambda x: x.split('\n')[0])
output_df = pd.read_csv('results/lwf_adapter_nolwf.csv')


# df = pd.read_csv('Data/artie-bias-corpus/artie-bias-corpus.tsv', delimiter = '\t')
# df['id'] = df['path'].apply(lambda x: x[:-4])
# output_df = pd.read_csv('artie-bias-corpus_transcriptions.csv')
result = pd.merge(output_df, df,   on='id', how="left")
# print(output_df, df)
language_counts = result["accent"].value_counts()

# languages_to_keep = language_counts[language_counts >=  30].index
# result = result[result['accent'].isin(languages_to_keep)]

eva = []
# Loop over each row in the dataset with tqdm for progress
for index, row in tqdm(result.iterrows(), total=len(result), desc="Processing CER & WER"):
    
    output = row['prediction']
    sentence= row['sentence']
    if len(output) >700:
        print(len(output), output)
        continue
    # Normalize the text (predictions and references) for each row (sentence)
    pred = [EnglishTextNormalizer()(output)]
    ref = [EnglishTextNormalizer()(sentence)]
    pred_token = pred[0].split() if isinstance(pred, list) else pred.split()
    ref_token = ref[0].split() if isinstance(ref, list) else ref.split()

    ember_hparams["wer_stats"].clear()
    ember_hparams["wer_stats"].append(
        ids=list(range(len([ref_token]))),
        predict=[pred_token],
        target=[ref_token],
    )
    ember_hparams["weighted_wer_stats"].clear()


    # Append the results for this sentence to the list
    eva.append({
        "id": row["id"],
        "WER": round(WER(pred,ref).item() * 100, 3),
        "CER": round(CER(pred,ref).item() * 100, 3),
        "MER": round(MER(pred,ref).item() * 100, 3),
        "WIL": round(WIL(pred,ref).item() * 100, 3),
        "Ember": round(ember_hparams["weighted_wer_stats"].summarize()["ember_wer"], 3),
        "SemDist": round(semdist.semdist(pred,ref), 5),
    })




results_df = pd.DataFrame(eva)
results_df.to_csv("results/eva_lwf_adapter_nolwf.csv", index=False)
print(results_df['SemDist'].max(), results_df['SemDist'].min())