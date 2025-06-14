import os
import torch
from data_utils import load_data
from train_models import (train_hybrid_model, walk_forward_patch_cv,
                          walk_forward_lstm_cv)
from model_io import save_models


START_DATE, END_DATE = '2020-01-01', '2025-06-14'
LSTM_START_DATE, LSTM_END_DATE = '2020-01-01', '2025-06-14'
TRAIN_SPLIT_RATIO, VAL_SPLIT_RATIO = 0.7, 0.2
CONTEXT_LENGTH, PREDICTION_LENGTH, PATCH_LENGTH = 30, 1, 10
WINDOW_LENGTH = CONTEXT_LENGTH - 5
NUM_TRAIN_EPOCHS_TST = 20
NUM_TRAIN_EPOCHS_LSTM = 50
BATCH_SIZE = 16
CV_SPLITS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_FOLDER = f'model_{END_DATE}'
MODEL_PATH = os.path.join('model', MODEL_FOLDER)


def main():
    data = load_data(
        START_DATE,
        END_DATE,
        LSTM_START_DATE,
        LSTM_END_DATE,
        TRAIN_SPLIT_RATIO,
        VAL_SPLIT_RATIO,
        CONTEXT_LENGTH,
        PREDICTION_LENGTH,
        WINDOW_LENGTH,
        MODEL_PATH,
    )

    patch_model = walk_forward_patch_cv(
        data.train_dataset,
        CV_SPLITS,
        CONTEXT_LENGTH,
        PREDICTION_LENGTH,
        PATCH_LENGTH,
        NUM_TRAIN_EPOCHS_TST,
        BATCH_SIZE,
        DEVICE,
    )

    lstm_model = walk_forward_lstm_cv(
        data.lstm_train_data,
        CV_SPLITS,
        NUM_TRAIN_EPOCHS_LSTM,
        BATCH_SIZE,
        DEVICE,
        WINDOW_LENGTH,
        PREDICTION_LENGTH,
    )

    hyb_model = train_hybrid_model(patch_model, lstm_model, data.val_dataset, DEVICE)

    save_models(MODEL_PATH, patch_model, lstm_model, hyb_model, data.scaler)


if __name__ == '__main__':
    main()
