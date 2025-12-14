import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressorHead(nn.Module):
    def __init__(self, hidden_size, output_dim=9, dropout=0.2):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, hidden_states):
        attn_scores = self.attention(hidden_states).squeeze(-1)  # (batch, time)
        attn_weights = F.softmax(attn_scores, dim=1)  # normalize

        pooled = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)

        return self.regressor(pooled)
