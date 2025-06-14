import torch
import torch.nn as nn
from transformers import PatchTSTForPrediction, PatchTSTConfig


class LSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 7, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :]).unsqueeze(-1)


def create_patch_model(context_length: int, prediction_length: int, patch_length: int, device: torch.device):
    config = PatchTSTConfig(
        context_length=context_length,
        prediction_length=prediction_length,
        patch_length=patch_length,
        num_input_channels=1,
    )
    return PatchTSTForPrediction(config).to(device)
