import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# ------------------------
# App Title
# ------------------------
st.title('Stockify 💹: Stock Prediction App')
st.text('Welcome to Stockify! Navigate the Future of Finance.')

# ------------------------
# Stock selection and prediction period
# ------------------------
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = ('AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 5)
period = n_years * 365

# ------------------------
# Load stock data
# ------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...Done!")

# ------------------------
# Display raw data
# ------------------------
st.subheader('Raw data')
st.write(data.tail())

# ------------------------
# Plot raw data
# ------------------------
# def plot_raw_data():
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
#     fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
#     fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
#     st.plotly_chart(fig)

# plot_raw_data()

# ------------------------
# Prepare data for Prophet safely
# ------------------------
# Flatten y to 1D Series to avoid TypeError
y = pd.Series(data['Close'].values.flatten())
y = pd.to_numeric(y, errors='coerce')

df_train = pd.DataFrame({
    'ds': pd.to_datetime(data['Date'], errors='coerce'),
    'y': y
})

# Drop rows with invalid values
df_train = df_train.dropna(subset=['ds', 'y'])

# Check if DataFrame is empty
if df_train.empty:
    st.error("DataFrame is empty after cleaning. Cannot fit Prophet model.")
else:
    # ------------------------
    # Fit Prophet model
    # ------------------------
    m = Prophet()
    m.fit(df_train)

    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # ------------------------
    # Display forecast
    # ------------------------
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.subheader('Forecast plot')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.subheader('Forecast components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)
