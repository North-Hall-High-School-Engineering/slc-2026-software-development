import torch.nn as nn

from models.regressor import RegressorHead


class CombinedModel(nn.Module):
    def __init__(self, hubert_model):
        super().__init__()
        self.hubert = hubert_model
        self.regressor = RegressorHead(hubert_model.config.hidden_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.hubert(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        preds = self.regressor(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_fn(preds, labels)
        return {"loss": loss, "logits": preds}
