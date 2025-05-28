import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from utils.Adapter import AccentAdapter


class Whisper(torch.nn.Module):
    def __init__(self, opt, processor ):
        super(Whisper, self).__init__()
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
        encoder_layer0 = self.model.model.get_encoder().layers[0]
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
    def set_eval_mode(self):
        # Set the entire model to evaluation mode
        self.model.eval()

        # Disable gradient calculation for LoRA layers specifically
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = False
    def generate(self, input_features, **kwargs):
        return self.model.generate(input_features=input_features, **kwargs)      
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