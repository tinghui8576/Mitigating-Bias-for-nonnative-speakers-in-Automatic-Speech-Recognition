import torch
import torch.nn as nn
import torch.nn.functional as F

# Neural
class AccentAdapter(nn.Module):
    def __init__(self, hidden_size, num_bases):
        super().__init__()
        self.num_bases = num_bases
        self.bias_bases = nn.Parameter(torch.randn(num_bases, hidden_size, requires_grad=True))
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_bases)
        )

    def forward(self, hidden_states, accent_id=None):
        pooled = hidden_states.mean(dim=1)  # (batch, hidden)
        gate_logits = self.gate_net(pooled)  # (batch, num_bases)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (batch, num_bases)

        # Weighted sum of bias bases (B * alpha)
        adapted_bias = torch.matmul(gate_weights, self.bias_bases)  # (batch, hidden)
        adapted_bias = adapted_bias.unsqueeze(1)  # (batch, 1, hidden)

        # Add adapted bias to each timestep
        output = hidden_states + adapted_bias
        
        # Optional classification loss if accent_id is provided
        if accent_id is not None:
            accent_id = accent_id.to(gate_logits.device)
            loss = F.cross_entropy(gate_logits, accent_id)
        
            return output, loss
        return output, None

# Fixed
# class AccentAdapter(nn.Module):
#     def __init__(self, hidden_size, num_bases):
#         super().__init__()
#         self.num_bases = num_bases
#         self.bias_bases = nn.Parameter(torch.randn(num_bases, hidden_size, requires_grad=True))

#     def forward(self, hidden_states, accent_id=None):
        
#         # Weighted sum of bias bases (B * alpha)
#         accent_one_hot = F.one_hot(accent_id, num_classes=self.num_bases).float()
#         # print(accent_one_hot)
#         adapted_bias = torch.matmul(accent_one_hot, self.bias_bases)  # (batch, hidden)
#         adapted_bias = adapted_bias.unsqueeze(1)  # (batch, 1, hidden)


#         # Add adapted bias to each timestep
#         output = hidden_states + adapted_bias
        
#         return output



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class AccentAdapter(nn.Module):
#     def __init__(self, hidden_size, num_bases, reg_weight=1e-4):
#         super().__init__()
#         self.num_bases = num_bases
#         self.reg_weight = reg_weight
#         self.bias_bases = nn.Parameter(torch.randn(num_bases, hidden_size, requires_grad=True))

#     def forward(self, hidden_states, accent_id=None):
        
#         # Weighted sum of bias bases (B * alpha)
#         accent_one_hot = F.one_hot(accent_id, num_classes=self.num_bases).float()
#         adapted_bias = torch.matmul(accent_one_hot, self.bias_bases)  # (batch, hidden)
#         adapted_bias = adapted_bias.unsqueeze(1)  # (batch, 1, hidden)

#         adapted_bias = adapted_bias - adapted_bias.mean(dim=-1, keepdim=True)

#         # Add adapted bias to each timestep
#         output = hidden_states + 0.5 * adapted_bias
        
#         reg_loss = self.reg_weight * (self.bias_bases ** 2).sum()
#         print("reg_loss requires grad?", reg_loss.requires_grad)

#         return output, reg_loss