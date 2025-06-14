import joblib
import pandas as pd
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from transformers import PatchTSTForPrediction, PatchTSTConfig, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Константы
START_DATE, END_DATE = '2020-01-01', '2025-05-16'
LSTM_START_DATE, LSTM_END_DATE = '2020-01-01', '2025-05-16'
TRAIN_SPLIT_RATIO, VAL_SPLIT_RATIO = 0.7, 0.2
CONTEXT_LENGTH, PREDICTION_LENGTH, PATCH_LENGTH = 30, 1, 10
WINDOW_LENGTH = CONTEXT_LENGTH - 5
NUM_TRAIN_EPOCHS_TST = 20
NUM_TRAIN_EPOCHS_LSTM = 50
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры Early Stopping
patience = 3
min_delta = 0.0001
best_loss = np.inf
epochs_no_improve = 0

# Для сохранения новой модели
model_folder = f'model_{END_DATE}'
model_path = os.path.join('model', model_folder)
os.makedirs(model_path, exist_ok=True)

# Датасеты
class CryptoDataset(Dataset):
    def __init__(self, data, context_length, prediction_length):
        self.data = data
        self.context_length = context_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.data) - self.context_length - self.prediction_length

    def __getitem__(self, idx):
        x = self.data['y_scaled'].values[idx:idx+self.context_length]
        y = self.data['y_scaled'].values[idx+self.context_length:idx+self.context_length+self.prediction_length]
        return {'past_values': torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
                'future_values': torch.tensor(y, dtype=torch.float32).unsqueeze(-1)}


class WindowedDataset(Dataset):
    def __init__(self, data, window_length, prediction_length):
        self.data = data['y_scaled'].values
        self.window_length = window_length
        self.prediction_length = prediction_length

    def __len__(self):
        return len(self.data) - self.window_length - self.prediction_length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_length]
        y = self.data[idx + self.window_length: idx + self.window_length + self.prediction_length]
        return {'past_values': torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
                'future_values': torch.tensor(y, dtype=torch.float32).unsqueeze(-1)}


# LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=7, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :]).unsqueeze(-1)


# Загрузка и подготовка данных
btc_data = yf.download('BTC-USD', start=START_DATE, end=END_DATE)
btc_data.reset_index(inplace=True)
btc_data.to_csv('model/btc_data_raw.csv', index=False)  # исходные данные без обработки
btc_data = pd.read_csv('model/btc_data_raw.csv')

# Переименовываем и выбираем только нужные столбцы
btc_data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
btc_data = btc_data[['ds', 'y']]  # оставляем только даты и цены закрытия

# Приведение типов и масштабирование
btc_data['ds'] = pd.to_datetime(btc_data['ds']).dt.strftime('%Y-%m-%d')
btc_data['y'] = pd.to_numeric(btc_data['y'], errors='coerce')

# Очистка от возможных NaN
btc_data.dropna(inplace=True)

# Масштабируем данные
scaler = MinMaxScaler()
btc_data['y_scaled'] = scaler.fit_transform(btc_data[['y']])

# сохраняем финальные обработанные данные
btc_data.to_csv(f'{model_path}/btc_data.csv', index=False)



# Разделение данных
train_size = int(len(btc_data) * TRAIN_SPLIT_RATIO)
val_size = int(len(btc_data) * VAL_SPLIT_RATIO)
train_data, val_data = btc_data.iloc[:train_size], btc_data.iloc[train_size:train_size+val_size]
test_data = btc_data.iloc[train_size+val_size:]

train_dataset = CryptoDataset(train_data, CONTEXT_LENGTH, PREDICTION_LENGTH)
val_dataset = CryptoDataset(val_data, CONTEXT_LENGTH, PREDICTION_LENGTH)
test_dataset = CryptoDataset(test_data, CONTEXT_LENGTH, PREDICTION_LENGTH)
lstm_train_data = btc_data[(btc_data['ds'] >= LSTM_START_DATE) & (btc_data['ds'] <= LSTM_END_DATE)]

# Transformer PatchTST
patch_config = PatchTSTConfig(context_length=CONTEXT_LENGTH,
                              prediction_length=PREDICTION_LENGTH,
                              patch_length=PATCH_LENGTH,
                              num_input_channels=1)
patch_model = PatchTSTForPrediction(patch_config).to(DEVICE)

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./results",
    # overwrite_output_dir=True,
    num_train_epochs=NUM_TRAIN_EPOCHS_TST,
    do_eval=True,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=10,
    label_names=["future_values"],
    logging_strategy="epoch",
    logging_dir="./logs/",)


early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,
    early_stopping_threshold=0.001,)

trainer = Trainer(model=patch_model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset,
                  callbacks=[early_stopping_callback])

# Transformer fine-tuning c EarlyStopping
trainer.train()


lstm_model = LSTM().to(DEVICE)
criterion, optimizer = nn.MSELoss(), torch.optim.Adam(lstm_model.parameters(), lr=0.001)
windowed_train_loader = DataLoader(
    WindowedDataset(lstm_train_data, WINDOW_LENGTH, PREDICTION_LENGTH),
    batch_size=BATCH_SIZE, shuffle=True)
windowed_val_loader = DataLoader(
    WindowedDataset(val_data, WINDOW_LENGTH, PREDICTION_LENGTH),
    batch_size=BATCH_SIZE, shuffle=False
)


# LSTM с EarlyStopping
lstm_model.train()

for epoch in range(NUM_TRAIN_EPOCHS_LSTM):
    epoch_loss = 0
    for batch in windowed_train_loader:
        optimizer.zero_grad()
        loss = criterion(lstm_model(batch['past_values'].to(DEVICE)), batch['future_values'].to(DEVICE))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(windowed_train_loader)

    # Валидационная проверка
    lstm_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in windowed_val_loader:
            val_preds = lstm_model(batch['past_values'].to(DEVICE))
            val_loss += criterion(val_preds, batch['future_values'].to(DEVICE)).item()

    val_loss /= len(windowed_val_loader)

    print(f"Epoch {epoch+1}/{NUM_TRAIN_EPOCHS_LSTM}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Проверка Early Stopping
    if best_loss - val_loss > min_delta:
        best_loss = val_loss
        epochs_no_improve = 0
        torch.save(lstm_model.state_dict(), 'best_lstm_model.pth')  # Сохраняем лучшую модель
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("Early stopping triggered!")
        lstm_model.load_state_dict(torch.load('best_lstm_model.pth'))  # Загружаем лучшую модель
        break

    lstm_model.train()


val_loader_hyb = DataLoader(val_dataset, batch_size=1)

X_hyb, y_hyb = [], []

patch_model.eval()
lstm_model.eval()

with torch.no_grad():
    for sample in val_loader_hyb:
        past_values = sample['past_values'].to(DEVICE)
        future_values = sample['future_values'].cpu().numpy().flatten()

        patch_pred = patch_model.generate(past_values).sequences.cpu().numpy().flatten()
        lstm_pred = lstm_model(past_values).cpu().numpy().flatten()

        X_hyb.append([patch_pred[0], lstm_pred[0]])
        y_hyb.append(future_values[0])

X_hyb = np.array(X_hyb)
y_hyb = np.array(y_hyb)

# Обучение гибрид-модели (линейная регрессия)
hyb_model = LinearRegression()
hyb_model.fit(X_hyb, y_hyb)


torch.save(patch_model.state_dict(), f'{model_path}/transformer_model.pth')
torch.save(lstm_model.state_dict(), f'{model_path}/lstm_model.pth')
joblib.dump(hyb_model, f'{model_path}/hyb_model.pkl')
joblib.dump(scaler, f'{model_path}/scaler.pkl')


# Прогноз на тесте
real_values, patch_predictions, lstm_predictions, hyb_predictions, forecast_dates = [], [], [], [], []

test_loader_hyb = DataLoader(test_dataset, batch_size=1)

with torch.no_grad():
    for idx, sample in enumerate(test_loader_hyb):
        past_values = sample['past_values'].to(DEVICE)
        real_future = sample['future_values'].cpu().numpy().flatten()

        patch_pred = patch_model.generate(past_values).sequences.cpu().numpy().flatten()
        lstm_pred = lstm_model(past_values).cpu().numpy().flatten()

        hyb_input = np.array([patch_pred, lstm_pred]).reshape(1, -1)
        hyb_pred = hyb_model.predict(hyb_input)

        real_values.extend(real_future)
        patch_predictions.extend(patch_pred)
        lstm_predictions.extend(lstm_pred)
        hyb_predictions.extend(hyb_pred)
        forecast_dates.append(pd.to_datetime(test_data['ds'].iloc[idx + CONTEXT_LENGTH]))

# Обратная нормализация
real_prices = scaler.inverse_transform(np.array(real_values).reshape(-1, 1)).flatten()
# patch_prices = scaler.inverse_transform(np.array(patch_predictions).reshape(-1, 1)).flatten()
# lstm_prices = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()
hyb_prices = scaler.inverse_transform(np.array(hyb_predictions).reshape(-1, 1)).flatten()

# Метрики
# models = {'Transformer': patch_prices, 'LSTM': lstm_prices, 'Hybrid-Model': hyb_prices}


mae = mean_absolute_error(real_prices, hyb_prices)
rmse = np.sqrt(mean_squared_error(real_prices, hyb_prices))
mape = np.mean(np.abs((real_prices - hyb_prices) / real_prices)) * 100
r2 = r2_score(real_prices, hyb_prices)

print(f"\nHybrid-Model Metrics:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R^2: {r2:.2f}")

# График прогнозов
plt.figure(figsize=(14, 7))
plt.plot(forecast_dates, real_prices, label='Реальные данные', color='black')
plt.plot(forecast_dates, hyb_prices, label='Гибридная модель прогноз', alpha=0.9, linestyle='--')

plt.xlabel('Дата')
plt.ylabel('Цена BTC (USD)')
plt.title('Прогнозы модели на тестовой выборке')
plt.legend()
plt.grid(True)
plt.show()



# авторегрессионная проверка
forecast_horizon = 7  # на сколько шагов вперед прогнозируем
initial_context = test_dataset[0]['past_values'].to(DEVICE).unsqueeze(0)

# patch_model.eval()
# lstm_model.eval()
hybrid_predictions = []

current_input = initial_context.clone()

with torch.no_grad():
    for step in range(forecast_horizon):
        # прогноз Transformer
        patch_pred = patch_model.generate(current_input).sequences.cpu().numpy().flatten()

        # прогноз LSTM
        lstm_pred = lstm_model(current_input).cpu().numpy().flatten()

        # мета-прогноз
        hyb_input = np.array([patch_pred[0], lstm_pred[0]]).reshape(1, -1)
        hyb_pred = hyb_model.predict(hyb_input)
        hybrid_predictions.append(hyb_pred[0])

        # новый вход — используем мета-прогноз
        new_input = np.append(current_input.cpu().numpy().flatten()[1:], hyb_pred)
        current_input = torch.tensor(new_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)

# Обратная нормализация
hybrid_predictions_real_scale = scaler.inverse_transform(np.array(hybrid_predictions).reshape(-1, 1)).flatten()

# Реальные значения
real_future_values = scaler.inverse_transform(
    test_data['y_scaled'].values[CONTEXT_LENGTH:CONTEXT_LENGTH + forecast_horizon].reshape(-1, 1)
).flatten()

# Метрики и график
mae_hybrid = mean_absolute_error(real_future_values, hybrid_predictions_real_scale)
rmse_hybrid = np.sqrt(mean_squared_error(real_future_values, hybrid_predictions_real_scale))

print(f"Авторегрессионный прогноз гибридной модели на {forecast_horizon} шагов:")
print(f"MAE: {mae_hybrid:.2f}, RMSE: {rmse_hybrid:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(range(forecast_horizon), real_future_values, label='Реальные значения')
plt.plot(range(forecast_horizon), hybrid_predictions_real_scale, label='Авторегрессионный прогноз Гибридной модели')
plt.xlabel('Шаг прогноза')
plt.ylabel('Цена BTC (USD)')
plt.title('Авторегрессионный прогноз Гибридной модели (Transformer + LSTM)')
plt.legend()
plt.grid(True)
plt.show()
