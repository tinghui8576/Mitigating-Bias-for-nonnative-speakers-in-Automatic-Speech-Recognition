
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F

class AdapterWrappedEncoderLayer(torch.nn.Module):
    def __init__(self, original_layer, hidden_size, num_bases, reg_weight=1e-4):
        super().__init__()
        self.original_layer = original_layer
        self._accent_id = None
        self._reg_loss = None

        self.num_bases = num_bases
        self.reg_weight = reg_weight
        init_std = 0.1
        self.bias_bases = torch.nn.Parameter(torch.randn(num_bases, hidden_size)* init_std, requires_grad=True)

    def set_accent_id(self, accent_id):
        self._accent_id = accent_id
 
    def forward(self, *args, **kwargs):
        output = self.original_layer(*args, **kwargs)
        hidden_states = output[0] if isinstance(output, tuple) else output
        batch_size = hidden_states.size(0)
        accent_ids = self._accent_id
        
        if accent_ids.size(0) != batch_size:
            accent_ids = accent_ids[:batch_size]
        # Weighted sum of bias bases (B * alpha)
        accent_one_hot = F.one_hot(self._accent_id, num_classes=self.num_bases).float()
        adapted_bias = torch.matmul(accent_one_hot, self.bias_bases)  # (batch, hidden)
        adapted_bias = adapted_bias.unsqueeze(1)  # (batch, 1, hidden)
        adapted_bias = adapted_bias - adapted_bias.mean(dim=-1, keepdim=True)
        
        # Add adapted bias to each timestep
        adapted = hidden_states + adapted_bias
        
        self._reg_loss = self.reg_weight * (self.bias_bases ** 2).sum()

        if isinstance(output, tuple):
            return (adapted,) + output[1:]
        else:
            return adapted

class Whisper(torch.nn.Module):
    def __init__(self, opt):
        super(Whisper, self).__init__()
        self.processor = WhisperProcessor.from_pretrained(opt["model_name"], language=opt['language'], task="transcribe")

        base_model = WhisperForConditionalGeneration.from_pretrained(opt["model_name"])
        # Need to call before using Lora
        base_model.enable_input_require_grads()

        encoder_layers = base_model.model.encoder.layers
        encoder_layers[11] = AdapterWrappedEncoderLayer(
            encoder_layers[11],
            hidden_size=base_model.config.d_model,
            num_bases=14
        )
        self._encoder_layers = (encoder_layers)
        if opt['strategy'].lower() == 'lwf':
            self._base_model = base_model
            
            N = 9  # number of top layers to freeze
            encoder_N = 6
            decoder_layers = self._base_model.model.decoder.layers  

            for i in range(0, N):  
                if i < encoder_N:
                    for param in encoder_layers[i].parameters():
                        param.requires_grad = False
                for param in decoder_layers[i].parameters():
                    param.requires_grad = False

        elif opt['strategy'].lower() == 'lora':
            lora_config = LoraConfig(
                r=32, 
                lora_alpha=64,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
            )
            self._base_model = get_peft_model(base_model, lora_config)
            for name, param  in self._base_model.named_parameters():
                if ("bias_bases" in name) or ("lora" in name):
                    param.requires_grad = True
        else:
            print("The Strategy Name is wrong. It should be either Lora or LwF")
        self.alpha_loss_weight = 0.1

        self._base_model.to(opt["device"])
        self.to(opt['device'])

        self._base_model.model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled:", self._base_model.model.is_gradient_checkpointing)
        
    # Add this method to make it compatible with Seq2SeqTrainer
    def gradient_checkpointing_enable(self, *args, **kwargs):
        """Enable gradient checkpointing for the internal model."""
        self._base_model.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self, *args, **kwargs):
        """Disable gradient checkpointing for the internal model."""
        self._base_model.model.gradient_checkpointing_disable()

    def forward(self, input_features, labels=None, accent_id=None, **kwargs):    
        self._encoder_layers[11].set_accent_id(accent_id)
        outputs = self._base_model(input_features=input_features, labels=labels)        
        reg_loss = self._encoder_layers[11]._reg_loss
        if reg_loss is not None:
            final_loss = outputs.loss + self.alpha_loss_weight * reg_loss
            outputs.loss = final_loss     
        return outputs
    
    def set_eval_mode(self):
        # Set the entire model to evaluation mode
        self._base_model.eval()

    def generate(self, input_features, accent_id=None, **kwargs):
        self._encoder_layers[11].set_accent_id(accent_id)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")
        outputs = self._base_model.generate(input_features=input_features, forced_decoder_ids=forced_decoder_ids)
        
        return outputs

    @property
    def config(self):
        return self._base_model.config

    @property
    def generation_config(self):
        return self._base_model.generation_config

    @generation_config.setter
    def generation_config(self, value):
        self._base_model.generation_config = value

    @property
    def device(self):
        return self._base_model.device

    @property
    def main_input_name(self):
        return self._base_model.main_input_name  
