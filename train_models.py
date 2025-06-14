import os
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from models import LSTM, create_patch_model
from data_utils import CryptoDataset, WindowedDataset


def train_patch_model(train_dataset: CryptoDataset, val_dataset: CryptoDataset,
                      context_length: int, prediction_length: int, patch_length: int,
                      num_epochs: int, batch_size: int, device: torch.device):
    patch_model = create_patch_model(context_length, prediction_length, patch_length, device)
    training_args = TrainingArguments(
        output_dir="./results_transformer",
        num_train_epochs=num_epochs,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        label_names=["future_values"],
        logging_strategy="epoch",
        logging_dir="./logs/",
    )
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,
        early_stopping_threshold=0.001,
    )
    trainer = Trainer(
        model=patch_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[early_stopping_callback],
    )
    trainer.train()
    return patch_model


def train_lstm_model(lstm_train_data: torch.utils.data.Dataset, val_data: torch.utils.data.Dataset,
                     num_epochs: int, batch_size: int, device: torch.device,
                     window_length: int, prediction_length: int,
                     patience: int = 3, min_delta: float = 0.0001) -> LSTM:
    lstm_model = LSTM().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    train_loader = DataLoader(WindowedDataset(lstm_train_data, window_length, prediction_length),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowedDataset(val_data, window_length, prediction_length),
                            batch_size=batch_size, shuffle=False)

    best_loss = np.inf
    epochs_no_improve = 0
    lstm_model.train()
    os.makedirs('results_lstm', exist_ok=True)
    best_model_path = os.path.join('results_lstm', 'best_lstm_model.pth')
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(lstm_model(batch['past_values'].to(device)), batch['future_values'].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        lstm_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                val_preds = lstm_model(batch['past_values'].to(device))
                val_loss += criterion(val_preds, batch['future_values'].to(device)).item()
        val_loss /= len(val_loader)

        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(lstm_model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            lstm_model.load_state_dict(torch.load(best_model_path))
            break
        lstm_model.train()
    return lstm_model


def train_hybrid_model(patch_model, lstm_model, val_dataset: CryptoDataset, device: torch.device) -> LinearRegression:
    val_loader = DataLoader(val_dataset, batch_size=1)
    X_hyb, y_hyb = [], []
    patch_model.eval()
    lstm_model.eval()
    with torch.no_grad():
        for sample in val_loader:
            past_values = sample['past_values'].to(device)
            future_values = sample['future_values'].cpu().numpy().flatten()
            patch_pred = patch_model.generate(past_values).sequences.cpu().numpy().flatten()
            lstm_pred = lstm_model(past_values).cpu().numpy().flatten()
            X_hyb.append([patch_pred[0], lstm_pred[0]])
            y_hyb.append(future_values[0])
    X_hyb = np.array(X_hyb)
    y_hyb = np.array(y_hyb)
    hyb_model = LinearRegression()
    hyb_model.fit(X_hyb, y_hyb)
    return hyb_model

def walk_forward_patch_cv(train_dataset: CryptoDataset, n_splits: int, context_length: int,
                          prediction_length: int, patch_length: int, num_epochs: int,
                          batch_size: int, device: torch.device):
    df = train_dataset.data.reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = None
    for train_idx, val_idx in tscv.split(df):
        train_ds = CryptoDataset(df.iloc[train_idx].reset_index(drop=True), context_length, prediction_length)
        val_ds = CryptoDataset(df.iloc[val_idx].reset_index(drop=True), context_length, prediction_length)
        model = train_patch_model(train_ds, val_ds, context_length, prediction_length,
                                 patch_length, num_epochs, batch_size, device)
    return model


def walk_forward_lstm_cv(train_df, n_splits: int, num_epochs: int, batch_size: int,
                         device: torch.device, window_length: int, prediction_length: int):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model = None
    for train_idx, val_idx in tscv.split(train_df):
        tr_df = train_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_df.iloc[val_idx].reset_index(drop=True)
        model = train_lstm_model(tr_df, val_df, num_epochs, batch_size, device,
                                 window_length, prediction_length)
    return model