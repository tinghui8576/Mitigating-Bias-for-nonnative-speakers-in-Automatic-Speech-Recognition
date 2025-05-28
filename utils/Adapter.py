import torch
import torch.nn as nn
import torch.nn.functional as F

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