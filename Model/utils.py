import torch 
import os
def add_leace(model, ckpt_dir):
    eraser_dir = ckpt_dir
    erasers = []
    layers_to_hook = [11]
    
    # Load erasers for these layers
    for i in layers_to_hook:
        path = os.path.join(eraser_dir, f"leace_eraser_layer{i}.pt")
        erasers.append(torch.load(path))

    # Register hooks only on these layers
    for layer_idx, eraser in zip(layers_to_hook, erasers):
        layer = model.model.encoder.layers[layer_idx]
        
        def make_hook(my_eraser):
            def hook_fn(module, input, output):
                hidden_states = output[0]
                erased = my_eraser(hidden_states.squeeze(0)).unsqueeze(0)
                return (erased,)
            return hook_fn
        
        layer.register_forward_hook(make_hook(eraser))
    return model