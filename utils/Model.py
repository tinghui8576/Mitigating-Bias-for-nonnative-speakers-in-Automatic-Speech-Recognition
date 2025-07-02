import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from utils.Adapter import AccentAdapter

import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from utils.Adapter import AccentAdapter
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
    def __init__(self, model_name, language, device = "cpu"):

        
        super(Whisper, self).__init__()

        self.processor = WhisperProcessor.from_pretrained(model_name, language=language, task="transcribe")
        basemodel = WhisperForConditionalGeneration.from_pretrained(model_name)
        self._base_model = basemodel  # Rename to avoid recursion

        self._base_model.to(device)
        self.to(device)

        encoder_layers = self._base_model.model.get_encoder().layers
        encoder_layers[11] = AdapterWrappedEncoderLayer(
            encoder_layers[11], self._base_model.config.d_model, num_bases=14
        )
        
        
        # Enable gradient checkpointing in the base model
        # basemodel.gradient_checkpointing_enable()
        self.alpha_loss_weight = 0.1

        N = 9  # number of top layers to freeze
        encoder_N = 6
        encoder_layers =self._base_model.model.encoder.layers  # list of encoder layers
        decoder_layers = self._base_model.model.decoder.layers  # list of decoder layers

        for i in range(0, N):  
            if i < encoder_N:
                for param in encoder_layers[i].parameters():
                    param.requires_grad = False
       
            for param in decoder_layers[i].parameters():
                param.requires_grad = False
        

        print("Gradient checkpointing enabled:", self._base_model.model.is_gradient_checkpointing)
    # Add this method to make it compatible with Seq2SeqTrainer
    def gradient_checkpointing_enable(self, *args, **kwargs):
        """Enable gradient checkpointing for the internal model."""
        self._base_model.model.gradient_checkpointing_enable()

    # Optionally, add the disable method as well
    def gradient_checkpointing_disable(self, *args, **kwargs):
        """Disable gradient checkpointing for the internal model."""
        self._base_model.model.gradient_checkpointing_disable()

    def forward(self, input_features, labels=None, accent_id=None, **kwargs):
        self._base_model.model.encoder.layers[11].set_accent_id(accent_id)
        outputs = self._base_model(input_features=input_features, labels=labels)
        reg_loss = self._base_model.model.encoder.layers[11]._reg_loss
        if reg_loss is not None:
            outputs.loss += self.alpha_loss_weight * reg_loss
        return outputs


    def set_eval_mode(self):
        # Set the entire model to evaluation mode
        self._base_model.eval()

        # Disable gradient calculation for LoRA layers specifically
        for name, param in self._base_model.named_parameters():
            if "lora" in name:
                param.requires_grad = False

    def generate(self, input_features, accent_id=None, **kwargs):
        self._base_model.model.encoder.layers[11].set_accent_id(accent_id)
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")
        outputs = self._base_model.generate(input_features=input_features, forced_decoder_ids=forced_decoder_ids, **kwargs)
        
        return outputs

    # def prepare_inputs_for_generation(self, *args, **kwargs):
    #     return self._base_model.prepare_inputs_for_generation(*args, **kwargs)

    # def resize_token_embeddings(self, *args, **kwargs):
    #     return self._base_model.resize_token_embeddings(*args, **kwargs)

    # def state_dict(self, *args, **kwargs):
    #     return self._base_model.state_dict(*args, **kwargs)

    # def load_state_dict(self, *args, **kwargs):
    #     return self._base_model.load_state_dict(*args, **kwargs)

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

    # def __getattr__(self, name):
    #     try:
    #         return super().__getattribute__(name)
    #     except AttributeError:
    #         base = super().__getattribute__('_base_model')
    #         return getattr(base, name)

    # def __getattr__(self, name):
    #     # Delegate missing attributes to _base_model
    #     try:
    #         return getattr(self._base_model, name)
    #     except AttributeError:
    #         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

class WhisperPEFT(torch.nn.Module):
    def __init__(self, opt, processor ):
        super(WhisperPEFT, self).__init__()
        self.model_name = opt["model_name"]
        
        self.processor = processor 
        basemodel = WhisperForConditionalGeneration.from_pretrained(self.model_name)

        # Enable gradient checkpointing in the base model
        # basemodel.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=32, 
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.model = get_peft_model(basemodel, lora_config)
        self.Biasadapter = AccentAdapter(self.model.config.d_model, 14)
        
        self.alpha_loss_weight = 0.1

        self.device = opt['device']
        self.model.to(self.device)

        # Store hook-related state
        self.adapter_runtime = {"loss": None}
        self._hook_handle = None
        self._current_accent_id = None  
        # Register persistent hook ONCE
        encoder_layer0 = self.model.model.get_encoder().layers[11]
        encoder_layer0.register_forward_hook(self._hook_fn)
        

        print("Gradient checkpointing enabled:", self.model.model.is_gradient_checkpointing)
    # Add this method to make it compatible with Seq2SeqTrainer
    def gradient_checkpointing_enable(self, *args, **kwargs):
        """Enable gradient checkpointing for the internal model."""
        self.model.model.gradient_checkpointing_enable()

    # Optionally, add the disable method as well
    def gradient_checkpointing_disable(self, *args, **kwargs):
        """Disable gradient checkpointing for the internal model."""
        self.model.model.gradient_checkpointing_disable()

    # ========================================
    # First version
    # def forward(self, input_features, labels=None, accent_id=None, **kwargs):
    #     # Apply adapter at the input level
    #     # adapted_input, alpha_loss = self.Biasadapter(accent_id, input_features)
        
    #     encoder_outputs = self.model.model.get_encoder()(input_features=input_features).last_hidden_state
    #     adapted_encoder, alpha_loss = self.Biasadapter(encoder_outputs, accent_id)
    #     # adapted_encoder = encoder_outputs

    #     outputs = self.model.model(
    #         encoder_outputs=(adapted_encoder,),
    #         labels=labels
    #     )
        
    #     total_loss = outputs.loss
    #     if alpha_loss is not None:
    #         total_loss = total_loss + self.alpha_loss_weight * alpha_loss

    #     return {
    #         "logits": outputs.logits,
    #         "loss": total_loss
    #     }


    # ========================================
    # Neural
    def _hook_fn(self, module, input, output):
        
        hidden_states = output[0] if isinstance(output, tuple) else output
        adapted_output, loss = self.Biasadapter(hidden_states, self._current_accent_id)
        
        self.adapter_runtime["loss"] = loss
        return (adapted_output,) + output[1:] if isinstance(output, tuple) else adapted_output
    
 
    def forward(self, input_features, labels=None, accent_id=None):
        # Apply adapter at the input level
        self._current_accent_id = accent_id  
        self.adapter_runtime["loss"] = None

        outputs = self.model.model(
            input_features=input_features,
            labels=labels
        )
        
        alpha_loss = self.adapter_runtime["loss"]
        total_loss = outputs.loss
        if alpha_loss is not None:
            total_loss += self.alpha_loss_weight * alpha_loss
        self._current_accent_id = None 
    
        return {
            "logits": outputs.logits,
            "loss": total_loss
        }
    def generate(self, input_features, **kwargs):
        return self.model.generate(input_features=input_features, **kwargs)     

    # ========================================
    # Fixed
    # def _hook_fn(self, module, input, output):
        
    #     hidden_states = output[0] if isinstance(output, tuple) else output
    #     adapted_output = self.Biasadapter(hidden_states, self._current_accent_id)
        
    #     return (adapted_output,) + output[1:] if isinstance(output, tuple) else adapted_output
    # def forward(self, input_features, labels=None, accent_id=None):
    #     # Apply adapter at the input level
    #     self._current_accent_id = accent_id  

    #     outputs = self.model.model(
    #         input_features=input_features,
    #         labels=labels
    #     )
        
    #     total_loss = outputs.loss
    #     self._current_accent_id = None 
    
    #     return {
    #         "logits": outputs.logits,
    #         "loss": total_loss,
    #     } 
    # def generate(self, input_features, accent_id=None, **kwargs):
    #     self._current_accent_id = accent_id
    #     outputs = self.model.generate(input_features=input_features, **kwargs)
    #     self._current_accent_id = None  # Reset after generation
    #     return outputs
    # def set_eval_mode(self):
    #     # Set the entire model to evaluation mode
    #     self.model.eval()

    #     # Disable gradient calculation for LoRA layers specifically
    #     for name, param in self.model.named_parameters():
    #         if "lora" in name:
    #             param.requires_grad = False
 
    # def predict(self, dataloader):
    #     self.model.eval()
    #     # set the forced ids
    #     outputs_dic = {}

    #     for input_features, file_ids, splits in tqdm(dataloader, desc="Evaluating WER"):
    #         forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en",task="transcribe")
    #         with torch.cuda.amp.autocast():  # Mixed precision
    #             predicted_ids = self.model.generate(input_features,forced_decoder_ids = forced_decoder_ids)
    #         with torch.no_grad():
    #             outputs = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

    #         # Store the result
    #         for (file_id, split, output) in zip(file_ids, splits, outputs):
    #             # Initialize the list if it's the first time seeing the file_id
    #             if file_id not in outputs_dic:
    #                 outputs_dic[file_id] = []
                
    #             # Append the split and transcription tuple
    #             outputs_dic[file_id].append((split, output))

    #     for file_id, transcriptions in outputs_dic.items():
    #         # Sort the transcriptions by split_start and split_end to concatenate in the correct order
    #         sorted_transcriptions = sorted(transcriptions, key=lambda x: x[0])  # Sort by split_start, split_end
    #         # Concatenate all transcriptions in the correct order
    #         full_transcription = "".join([transcription for _, transcription in sorted_transcriptions])
            
    #         # Store the combined transcription for the file_id
    #         outputs_dic[file_id] = full_transcription
    #     output_list = [{'id': key, 'output': value} for key, value in outputs_dic.items()]
    #     # Create a DataFrame
    #     outputs_df = pd.DataFrame(output_list)

    #     return outputs_df
