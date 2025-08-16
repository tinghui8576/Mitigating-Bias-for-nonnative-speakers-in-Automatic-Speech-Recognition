
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

def metrics(output, ground_truth):
    """
    Calculate WER, WIL, CER, MER, Ember and SemDist for the given predictions and references.
    """
    pred = [EnglishTextNormalizer()(output)]
    ref = [EnglishTextNormalizer()(ground_truth)]
    pred_token = pred[0].split() if isinstance(pred, list) else pred.split()
    ref_token = ref[0].split() if isinstance(ref, list) else ref.split()
    ember_hparams["wer_stats"].clear()
    ember_hparams["wer_stats"].append(
        ids=list(range(len([ref_token]))),
        predict=[pred_token],
        target=[ref_token],
    )
    ember_hparams["weighted_wer_stats"].clear()

    return {
        "WER": round(WER(pred,ref).item() * 100, 3),
        "CER": round(CER(pred,ref).item() * 100, 3),
        "MER": round(MER(pred,ref).item() * 100, 3),
        "WIL": round(WIL(pred,ref).item() * 100, 3),
        "Ember": round(ember_hparams["weighted_wer_stats"].summarize()["ember_wer"], 3),
        "SemDist": round(semdist.semdist(pred,ref), 5),
    }