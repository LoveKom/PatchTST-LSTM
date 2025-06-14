# PatchTST-LSTM: Гибридная модель для прогнозирования криптовалюты

Проект сочетает возможности моделей Transformer (PatchTST) и LSTM с графическим интерфейсом для прогнозирования временных рядов (Bitcoin).

## 📁 Содержимое проекта

- `transformer_lstm.py` — основной файл, запускающий обучение и оценку моделей.
- `data_utils.py` — загрузка и подготовка данных.
- `models.py` — определения и конфигурации моделей.
- `train_models.py` — обучение моделей и линейной регрессии.
- `model_io.py` — сохранение и загрузка обученных моделей.
- `metrics_calc.py` — вычисление метрик качества.
- `gui_predict.py` — графический интерфейс на Tkinter для загрузки моделей и визуального прогноза.

## 🔧 Установка

```bash
git clone https://github.com/yourusername/PatchTST-LSTM.git
cd PatchTST-LSTM
pip install -r requirements.txt
```
