
import streamlit as st
import yfinance as yf
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

nltk.download("vader_lexicon")

st.title("📈 NextMove - AI Stock Predictor")

ticker = st.text_input("Enter NSE Stock Symbol (e.g., TATAMOTORS.NS):")

if ticker:
    df = yf.download(ticker, start="2023-01-01")[["Close"]].dropna()

    newsapi = NewsApiClient(api_key=st.secrets["NEWSAPI_KEY"])
    articles = newsapi.get_everything(q=ticker.split(".")[0], language="en", page_size=10)
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(a["title"])["compound"] for a in articles["articles"]]
    sentiment = np.mean(scores) if scores else 0
    st.metric("Average News Sentiment", f"{sentiment:.2f}")
    df["Sentiment"] = sentiment

    st.line_chart(df["Close"])

    # Prepare LSTM data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[["Close", "Sentiment"]])
    seq_len = 60
    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i-seq_len:i])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, 2)),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Forecast next 30 days
    last_seq = data_scaled[-seq_len:].reshape(1, seq_len, 2)
    preds = []
    for _ in range(30):
        pred = model.predict(last_seq, verbose=0)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[:, 1:, :], [[[pred, data_scaled[-1, 1]]]], axis=1)

    dummy = np.zeros((30, 2))
    dummy[:, 0] = preds
    forecast = scaler.inverse_transform(dummy)[:, 0]
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30)
    forecast_df = pd.DataFrame({"Forecast": forecast}, index=future_dates)

    st.subheader("📊 30-Day Forecast")
    st.line_chart(forecast_df)
