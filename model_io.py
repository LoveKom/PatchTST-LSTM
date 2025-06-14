import os
import joblib
import torch
from typing import Tuple

from models import LSTM, create_patch_model


def save_models(path: str, patch_model, lstm_model, hyb_model, scaler) -> None:
    os.makedirs(path, exist_ok=True)
    torch.save(patch_model.state_dict(), os.path.join(path, 'transformer_model.pth'))
    torch.save(lstm_model.state_dict(), os.path.join(path, 'lstm_model.pth'))
    joblib.dump(hyb_model, os.path.join(path, 'hyb_model.pkl'))
    joblib.dump(scaler, os.path.join(path, 'scaler.pkl'))


def load_models(path: str, device: torch.device, context_length: int,
                prediction_length: int, patch_length: int) -> Tuple:
    patch_model = create_patch_model(context_length, prediction_length, patch_length, device)
    patch_model.load_state_dict(torch.load(os.path.join(path, 'transformer_model.pth'), map_location=device))
    lstm_model = LSTM().to(device)
    lstm_model.load_state_dict(torch.load(os.path.join(path, 'lstm_model.pth'), map_location=device))
    hyb_model = joblib.load(os.path.join(path, 'hyb_model.pkl'))
    scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
    return patch_model, lstm_model, hyb_model, scaler
