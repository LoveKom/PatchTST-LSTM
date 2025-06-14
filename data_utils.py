import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class CryptoDataset(Dataset):
    def __init__(self, data: pd.DataFrame, context_length: int, prediction_length: int):
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self) -> int:
        return len(self.data) - self.context_length - self.prediction_length

    def __getitem__(self, idx: int):
        x = self.data['y_scaled'].values[idx:idx + self.context_length]
        y = self.data['y_scaled'].values[idx + self.context_length:idx + self.context_length + self.prediction_length]
        return {
            'past_values': torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
            'future_values': torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        }


class WindowedDataset(Dataset):
    def __init__(self, data: pd.DataFrame, window_length: int, prediction_length: int):
        self.data = data['y_scaled'].values
        self.window_length = window_length
        self.prediction_length = prediction_length

    def __len__(self) -> int:
        return len(self.data) - self.window_length - self.prediction_length

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.window_length]
        y = self.data[idx + self.window_length: idx + self.window_length + self.prediction_length]
        return {
            'past_values': torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
            'future_values': torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        }


@dataclass
class DataSplit:
    train_dataset: CryptoDataset
    val_dataset: CryptoDataset
    test_dataset: CryptoDataset
    lstm_train_data: pd.DataFrame
    scaler: MinMaxScaler
    raw_data: pd.DataFrame
    test_data: pd.DataFrame


def load_data(start_date: str, end_date: str, lstm_start: str, lstm_end: str,
              train_split_ratio: float, val_split_ratio: float,
              context_length: int, prediction_length: int, window_length: int,
              path: str) -> DataSplit:
    btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
    btc_data.reset_index(inplace=True)
    os.makedirs(path, exist_ok=True)
    btc_data.to_csv(os.path.join(path, 'btc_data_raw.csv'), index=False)

    btc_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    btc_data = btc_data[['ds', 'y']]
    btc_data['ds'] = pd.to_datetime(btc_data['ds']).dt.strftime('%Y-%m-%d')
    btc_data['y'] = pd.to_numeric(btc_data['y'], errors='coerce')
    btc_data.dropna(inplace=True)

    scaler = MinMaxScaler()
    btc_data['y_scaled'] = scaler.fit_transform(btc_data[['y']])
    btc_data.to_csv(os.path.join(path, 'btc_data.csv'), index=False)

    train_size = int(len(btc_data) * train_split_ratio)
    val_size = int(len(btc_data) * val_split_ratio)

    train_data = btc_data.iloc[:train_size]
    val_data = btc_data.iloc[train_size:train_size + val_size]
    test_data = btc_data.iloc[train_size + val_size:]

    train_dataset = CryptoDataset(train_data, context_length, prediction_length)
    val_dataset = CryptoDataset(val_data, context_length, prediction_length)
    test_dataset = CryptoDataset(test_data, context_length, prediction_length)

    lstm_train_data = btc_data[(btc_data['ds'] >= lstm_start) & (btc_data['ds'] <= lstm_end)]

    return DataSplit(train_dataset, val_dataset, test_dataset, lstm_train_data, scaler, btc_data, test_data)
