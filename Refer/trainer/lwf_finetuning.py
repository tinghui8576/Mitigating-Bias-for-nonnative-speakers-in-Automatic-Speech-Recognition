
import torch.nn.functional as F
import torch
from transformers import Seq2SeqTrainer
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments

class LwFTrainingArguments(Seq2SeqTrainingArguments):
    """
    Training arguments for Learning without Forgetting (LwF).
    Includes alpha and temperature for distillation loss, and path to load old model.
    """
    def __init__(self,
                 alpha_lwf: float = 0.5,
                 temp_lwf: float = 2.0,
                 old_model_name: str = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_lwf = alpha_lwf
        self.temp_lwf = temp_lwf
        self.old_model_name = old_model_name



class LwFTrainer(Seq2SeqTrainer):
    """
    Trainer class for fine-tuning with Learning without Forgetting (LwF).
    Should be used with `args=LwFTrainingArguments`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load the old model from path if provided
        if hasattr(self.args, 'old_model_name') and self.args.old_model_name:

            device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
            
            self.old_model = WhisperForConditionalGeneration.from_pretrained(self.args.old_model_name).to(device)
            for param in self.old_model.parameters():
                param.requires_grad = False
            self.old_model.eval()
        else:
            self.old_model = None

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None  # Required for compatibility
    ):
        outputs = model(**inputs)
        new_logits = outputs.logits
        new_loss = outputs.loss
        labels = inputs["labels"]
        

        if self.old_model:
            old_inputs = inputs.copy()
            old_inputs.pop("accent_id", None)
            
            with torch.no_grad():
                old_outputs = self.old_model(**old_inputs)

            # Extract logits from both models
            old_logits = old_outputs.logits

            T = self.args.temp_lwf
            alpha = self.args.alpha_lwf

            # Make sure both logits have same shape
            if new_logits.shape == old_logits.shape:
                p_old = F.softmax(old_logits / T, dim=-1)
                p_new = F.log_softmax(new_logits / T, dim=-1)
                distill_loss = F.kl_div(p_new, p_old, reduction="batchmean") * (T * T)
                loss = new_loss + alpha * distill_loss
        return (loss, outputs) if return_outputs else loss