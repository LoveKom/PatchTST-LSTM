import tkinter as tk
import matplotlib
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import torch
import joblib
import pandas as pd
import numpy as np
from tkinter import ttk, filedialog, messagebox
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from transformers import PatchTSTForPrediction, PatchTSTConfig
from datetime import datetime, timedelta
import threading
import importlib
import os

matplotlib.use('TkAgg')
START_DATE, END_DATE = '2020-01-01', '2025-05-01'
LSTM_START_DATE, LSTM_END_DATE = '2020-01-01', '2025-05-01'
TRAIN_SPLIT_RATIO, VAL_SPLIT_RATIO = 0.7, 0.2
CONTEXT_LENGTH, PREDICTION_LENGTH, PATCH_LENGTH = 30, 1, 10
WINDOW_LENGTH = CONTEXT_LENGTH - 5
NUM_TRAIN_EPOCHS_TST = 20
NUM_TRAIN_EPOCHS_LSTM = 50
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

patch_config = PatchTSTConfig(context_length=CONTEXT_LENGTH,
                              prediction_length=PREDICTION_LENGTH,
                              patch_length=PATCH_LENGTH,
                              num_input_channels=1)

# LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=7, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :]).unsqueeze(-1)


def plot_data():
    global btc_data

    ax.clear()

    # Размер выборок
    train_size = int(len(btc_data) * TRAIN_SPLIT_RATIO)
    val_size = int(len(btc_data) * VAL_SPLIT_RATIO)
    test_start_idx = train_size + val_size

    # Реальные данные Train/Val
    ax.plot(btc_data['ds'][:test_start_idx], btc_data['y'][:test_start_idx],
            label='Train/Val данные', linewidth=2)

    # Реальные данные Test (включая последнее значение)
    ax.plot(btc_data['ds'][test_start_idx:], btc_data['y'][test_start_idx:],
            label='Test данные', linewidth=2)

    # Прогноз модели
    predictions = []

    context_length = CONTEXT_LENGTH
    for i in range(test_start_idx, len(btc_data)):
        input_window = btc_data['y_scaled'].values[i - context_length:i]
        current_input = torch.tensor(input_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)

        patch_model.eval()
        lstm_model.eval()

        with torch.no_grad():
            patch_pred = patch_model.generate(current_input).sequences.cpu().numpy().flatten()
            lstm_pred = lstm_model(current_input).cpu().numpy().flatten()
            hyb_input = np.array([patch_pred[0], lstm_pred[0]]).reshape(1, -1)
            hyb_pred = hyb_model.predict(hyb_input)

            predictions.append(hyb_pred[0])

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Исправленная часть: даты теперь ровно соответствуют прогнозам
    prediction_dates = btc_data['ds'][test_start_idx:].tolist()

    ax.plot(prediction_dates, predictions,
            '--', label='Прогноз модели', linewidth=2)

    ax.set_xlabel('Дата')
    ax.set_ylabel('Цена BTC (USD)')
    ax.set_title('Прогнозирование курса криптовалюты')
    ax.legend()
    ax.grid(True)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    def format_coord(x, y):
        date = mdates.num2date(x).strftime('%Y-%m-%d')
        return f"Дата: {date}, Цена: ${y:,.2f}"

    ax.format_coord = format_coord

    canvas.draw()

    # Метрики
    real_prices = btc_data['y'][test_start_idx:].values
    mae = mean_absolute_error(real_prices, predictions)
    rmse = np.sqrt(mean_squared_error(real_prices, predictions))
    mape = np.mean(np.abs((real_prices - predictions) / real_prices)) * 100
    r2 = r2_score(real_prices, predictions)

    metrics_text = f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R²: {r2:.2f}"
    metrics_label.config(text=metrics_text)


def on_closing():
    plt.close(fig)
    root.destroy()


def load_model():
    model_dir = filedialog.askdirectory(title="Выберите папку с моделью")
    if not model_dir:
        return

    hide_progress_bar()

    try:
        global patch_model, lstm_model, hyb_model, scaler, btc_data

        # Загрузка Transformer (PatchTST) модели
        patch_model = PatchTSTForPrediction(patch_config).to(DEVICE)
        patch_model.load_state_dict(torch.load(f'{model_dir}/transformer_model.pth', map_location=DEVICE))
        patch_model.eval()

        # Загрузка LSTM модели
        lstm_model = LSTM().to(DEVICE)
        lstm_model.load_state_dict(torch.load(f'{model_dir}/lstm_model.pth', map_location=DEVICE))
        lstm_model.eval()

        # Загрузка гибридной модели (Linear Regression) и scaler
        hyb_model = joblib.load(f'{model_dir}/hyb_model.pkl')
        scaler = joblib.load(f'{model_dir}/scaler.pkl')

        # Загрузка корректно предварительно обработанных данных
        # winsow_cool.py (правильная загрузка файла)
        btc_data = pd.read_csv(f'{model_dir}/btc_data.csv', parse_dates=['ds'])

        messagebox.showinfo("Загрузка", "Модель и данные успешно загружены!")

        # Построение графика и вывод метрик
        plot_data()

        # Разблокировка кнопки "Получить прогноз"
        btn_forecast.config(state=tk.NORMAL)

    except Exception as e:
        messagebox.showerror("Ошибка загрузки", str(e))

def train_model():
    train_window = tk.Toplevel(root)
    train_window.title("Обучить модель")

    tk.Label(train_window, text="Дата начала (YYYY-MM-DD):").grid(row=0, column=0, padx=5, pady=5)
    start_entry = tk.Entry(train_window)
    start_entry.grid(row=0, column=1, padx=5, pady=5)

    tk.Label(train_window, text="Дата окончания (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5)
    end_entry = tk.Entry(train_window)
    end_entry.grid(row=1, column=1, padx=5, pady=5)

    default_end = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    default_start = (datetime.today() - timedelta(days=1) - timedelta(days=365*5)).strftime('%Y-%m-%d')
    start_entry.insert(0, default_start)
    end_entry.insert(0, default_end)

    def start_training():
        s = start_entry.get()
        e = end_entry.get()
        train_window.destroy()
        threading.Thread(target=run_training, args=(s, e), daemon=True).start()

    tk.Button(train_window, text="Начать обучение", command=start_training).grid(row=2, column=0, columnspan=2, pady=10)


def run_training(start_date: str, end_date: str):
    root.after(0, show_progress_bar)
    root.after(0, log_progress, f"Старт обучения с {start_date} по {end_date}...")
    try:
        import transformer_lstm
        importlib.reload(transformer_lstm)

        transformer_lstm.START_DATE = start_date
        transformer_lstm.END_DATE = end_date
        transformer_lstm.LSTM_START_DATE = start_date
        transformer_lstm.LSTM_END_DATE = end_date
        transformer_lstm.MODEL_FOLDER = f"model_{end_date}"
        transformer_lstm.MODEL_PATH = os.path.join('model', transformer_lstm.MODEL_FOLDER)

        transformer_lstm.main()
        root.after(0, log_progress, "Обучение завершено.")
        root.after(0, hide_progress_bar)
        messagebox.showinfo("Обучение", "Обучение успешно завершено")
    except Exception as exc:
        root.after(0, log_progress, f"Ошибка обучения: {exc}")
        root.after(0, hide_progress_bar)
        messagebox.showerror("Ошибка", str(exc))


def forecast_autoregressive():
    forecast_horizon = int(forecast_spinbox.get())
    context_length = CONTEXT_LENGTH

    plot_data()

    initial_context = btc_data['y_scaled'].values[-context_length:]
    current_input = torch.tensor(initial_context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)

    hybrid_predictions = []

    patch_model.eval()
    lstm_model.eval()

    with torch.no_grad():
        for _ in range(forecast_horizon):
            patch_pred = patch_model.generate(current_input).sequences.cpu().numpy().flatten()
            lstm_pred = lstm_model(current_input).cpu().numpy().flatten()
            hyb_input = np.array([patch_pred[0], lstm_pred[0]]).reshape(1, -1)
            hyb_pred = hyb_model.predict(hyb_input)
            hybrid_predictions.append(hyb_pred[0])

            new_input = np.append(current_input.cpu().numpy().flatten()[1:], hyb_pred)
            current_input = torch.tensor(new_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)

    hybrid_predictions_real_scale = scaler.inverse_transform(np.array(hybrid_predictions).reshape(-1, 1)).flatten()

    last_date = btc_data['ds'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

    ax.plot(forecast_dates, hybrid_predictions_real_scale, label='Авторегрессионный прогноз',
            linestyle='--', linewidth=2, color='red')

    ax.legend()
    canvas.draw()

    # Очистка и заполнение таблицы
    forecast_table.delete(*forecast_table.get_children())

    for date, prediction in zip(forecast_dates, hybrid_predictions_real_scale):
        forecast_table.insert('', 'end', values=(date.strftime('%Y-%m-%d'), f"${prediction:,.2f}"))




# Создание главного окна
root = tk.Tk()
root.title("Гибридная модель Transformer-LSTM")
root.geometry("1200x700")

# Создание верхнего меню
menu_bar = tk.Menu(root)

file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Загрузить модель", command=load_model)
file_menu.add_command(label="Обучить модель", command=train_model)
file_menu.add_separator()
file_menu.add_command(label="Выход", command=root.quit)
menu_bar.add_cascade(label="Файл", menu=file_menu)

root.config(menu=menu_bar)

# Общий фрейм для графика и таблицы
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Фрейм для графика (слева)
frame_graph = tk.Frame(main_frame)
frame_graph.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Фрейм для таблицы (справа)
frame_table = tk.Frame(main_frame, width=300)
frame_table.pack(side=tk.RIGHT, fill=tk.Y)

forecast_table = ttk.Treeview(frame_table, columns=('date', 'forecast'), show='headings')
forecast_table.heading('date', text='Дата')
forecast_table.heading('forecast', text='Прогнозная цена (USD)')
forecast_table.column('date', anchor='center', width=120)
forecast_table.column('forecast', anchor='center', width=150)

forecast_table.pack(fill=tk.BOTH, expand=True, pady=10)

# Scrollbar для таблицы
table_scrollbar = ttk.Scrollbar(frame_table, orient='vertical', command=forecast_table.yview)
forecast_table.configure(yscrollcommand=table_scrollbar.set)
table_scrollbar.pack(side='right', fill='y')


# Заглушка графика
fig, ax = plt.subplots(figsize=(8, 4))
ax.text(0.5, 0.5, 'Добро пожаловать!', ha='center', va='center', fontsize=15, color='gray')
ax.set_xticks([])
ax.set_yticks([])

canvas = FigureCanvasTkAgg(fig, master=frame_graph)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Панель навигации Matplotlib
toolbar = NavigationToolbar2Tk(canvas, frame_graph)
toolbar.update()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Нижний фрейм для кнопок и управления
control_frame = tk.Frame(root)
control_frame.pack(fill=tk.X, padx=10, pady=5)

btn_forecast = tk.Button(control_frame, text="Получить прогноз", state=tk.DISABLED, command=forecast_autoregressive)
btn_forecast.pack(side=tk.LEFT, padx=5)

forecast_label = tk.Label(control_frame, text="Горизонт прогноза:")
forecast_label.pack(side=tk.LEFT, padx=5)

forecast_spinbox = tk.Spinbox(control_frame, from_=1, to=30, width=5)
forecast_spinbox.pack(side=tk.LEFT, padx=5)

# Метрики
metrics_label = tk.Label(root, text="Пожалуйста, загрузите модель.", font=("Arial", 12), pady=10)
metrics_label.pack(fill=tk.X, padx=10)

# Область для прогресса обучения с прокруткой
progress_frame = tk.Frame(root)
progress_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)

progress_text = tk.Text(progress_frame, height=10, state='disabled')
progress_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(progress_frame, command=progress_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
progress_text['yscrollcommand'] = scrollbar.set

# Progress bar at the very bottom. Hidden by default
progress_bar = ttk.Progressbar(root, mode='indeterminate')
progress_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
progress_bar.pack_forget()

def log_progress(message: str):
    progress_text.configure(state='normal')
    progress_text.insert(tk.END, message + '\n')
    progress_text.see(tk.END)
    progress_text.configure(state='disabled')

def show_progress_bar():
    progress_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
    progress_bar.start()


def hide_progress_bar():
    progress_bar.stop()
    progress_bar.pack_forget()

# Запуск окна приложения
root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
