import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# 1. Загрузка данных из CSV

def load_data(file_path):
    data = pd.read_csv(file_path, sep=';', encoding="windows-1251")
    return data

# 2. Подготовка данных
def prepare_data(data):
    # Преобразуем жанры в числовые метки
    label_encoder = LabelEncoder()
    data['genre_encoded'] = label_encoder.fit_transform(data['genre'])

    # Создаем входные и выходные данные
    X = data['year'].values.reshape(-1, 1)  # Признак: год
    y = data['genre_encoded'].values  # Цель: жанр

    # Нормализация года
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Преобразуем y в one-hot encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_onehot = onehot_encoder.fit_transform(y.reshape(-1, 1))

    return X, y_onehot, label_encoder, onehot_encoder, scaler

# 3. Создание модели нейронной сети с LSTM
def create_lstm_model(input_shape, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, activation='tanh'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(output_dim, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 4. Визуализация результатов

def plot_genre_trends_interactive(years, genres, predictions, label_encoder):
    unique_years = np.unique(years)
    genre_trends = {genre: [] for genre in label_encoder.classes_}

    for year in unique_years:
        year_indices = np.where(years == year)
        year_predictions = predictions[year_indices]
        genre_distribution = np.mean(year_predictions, axis=0)

        for i, genre in enumerate(label_encoder.classes_):
            genre_trends[genre].append(genre_distribution[i])

    # Создаем интерактивный график
    fig = go.Figure()

    for genre, trend in genre_trends.items():
        fig.add_trace(go.Scatter(
            x=unique_years,
            y=trend,
            mode='lines',
            name=genre,
            hoverinfo='x+y+name',
            visible='legendonly'  # Скрываем жанр по умолчанию
        ))

    fig.update_layout(
        title='Глобальный прогноз',
        xaxis_title='Годы',
        yaxis_title='Популярность',
        xaxis=dict(
            tickmode='linear',  # Линейное отображение меток
            tick0=unique_years[0],  # Начало отсчета
            dtick=2  # Шаг между метками
        ),
        legend_title='Genres',
        legend=dict(orientation='h', yanchor='bottom', y=-1.0, xanchor='center', x=0.5),
        template='plotly_white'
    )

    fig.show()

# Основной скрипт
if __name__ == "__main__":
    file_path = "Base1.csv"

    # Шаг 1: Загрузка данных
    data = load_data(file_path)

    # Шаг 2: Подготовка данных
    X, y_onehot, label_encoder, onehot_encoder, scaler = prepare_data(data)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    # Изменение формы входных данных для LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Шаг 3: Создание и обучение модели
    model = create_lstm_model(input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), output_dim=y_onehot.shape[1])
    model.fit(X_train_lstm, y_train, epochs=70, batch_size=16, validation_data=(X_test_lstm, y_test))

    # Шаг 4: Прогноз и визуализация
    X_lstm = X.reshape((X.shape[0], X.shape[1], 1))  # Изменение формы для всей выборки
    predictions = model.predict(X_lstm)

    # Шаг 5: Прогнозирование будущих годов
    future_years = np.arange(2023, 2031).reshape(-1, 1)  # Года с 2023 по 2030
    future_years_scaled = scaler.transform(future_years)  # Масштабируем годаФ
    future_years_lstm = future_years_scaled.reshape((future_years_scaled.shape[0], future_years_scaled.shape[1], 1))
    future_predictions = model.predict(future_years_lstm)  # Прогнозируем

    # Визуализация объединенных данных
    combined_years = np.concatenate([data['year'].values, future_years.flatten()])
    combined_predictions = np.concatenate([predictions, future_predictions])
    plot_genre_trends_interactive(combined_years, data['genre'].values, combined_predictions, label_encoder)
