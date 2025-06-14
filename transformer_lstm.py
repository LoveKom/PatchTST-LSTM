import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from data_utils import load_data
# from train_models import train_patch_model, train_lstm_model, train_hybrid_model
from train_models import (train_hybrid_model, walk_forward_patch_cv,
                          walk_forward_lstm_cv)
from model_io import save_models
from metrics_calc import compute_metrics


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

    # Evaluation
    test_loader = torch.utils.data.DataLoader(data.test_dataset, batch_size=1)
    real_values, hyb_predictions, forecast_dates = [], [], []

    patch_model.eval()
    lstm_model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(test_loader):
            past_values = sample['past_values'].to(DEVICE)
            real_future = sample['future_values'].cpu().numpy().flatten()
            patch_pred = patch_model.generate(past_values).sequences.cpu().numpy().flatten()
            lstm_pred = lstm_model(past_values).cpu().numpy().flatten()
            hyb_input = np.array([patch_pred, lstm_pred]).reshape(1, -1)
            hyb_pred = hyb_model.predict(hyb_input)
            real_values.extend(real_future)
            hyb_predictions.extend(hyb_pred)
            forecast_dates.append(data.test_data['ds'].iloc[idx + CONTEXT_LENGTH])

    forecast_dates = pd.to_datetime(forecast_dates)
    real_prices = data.scaler.inverse_transform(np.array(real_values).reshape(-1, 1)).flatten()
    hyb_prices = data.scaler.inverse_transform(np.array(hyb_predictions).reshape(-1, 1)).flatten()

    mae, rmse, mape, r2 = compute_metrics(real_prices, hyb_prices)

    print("\nHybrid-Model Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R^2: {r2:.2f}")

    plt.figure(figsize=(14, 7))
    plt.plot(forecast_dates, real_prices, label='Реальные данные', color='black')
    plt.plot(forecast_dates, hyb_prices, label='Гибридная модель прогноз', alpha=0.9, linestyle='--')
    plt.xlabel('Дата')
    plt.ylabel('Цена BTC (USD)')
    plt.title('Прогнозы модели на тестовой выборке')
    plt.legend()
    plt.grid(True)

    ax = plt.gca()
    # Установить формат даты и интервал меток (например, каждые 7 дней)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # Повернуть подписи
    plt.xticks(rotation=45)
    # Сделать сетку реже или отключить её по X
    ax.grid(True, which='major', axis='y')  # только по оси Y

    plt.show()


if __name__ == '__main__':
    main()
