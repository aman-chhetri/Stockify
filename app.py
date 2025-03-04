import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stockify 💹: Stock Prediction App')
st.text('Welcome to Stockify! Navigate the Future of Finance.')

# Set the stock ticker symbols
stocks = ('AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Set the number of years for prediction
n_years = st.slider('Years of prediction:', 1, 5)
period = n_years * 365

# Load the stock data using yfinance
@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        if data.empty:
            st.error("Error: No data found for the selected stock.")
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data
data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... Done!")

if data is None or data.empty:
    st.stop()

# Display the raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Ensure data contains the necessary columns
if 'Date' not in data.columns or 'Close' not in data.columns:
    st.error("Error: Required columns ('Date', 'Close') are missing in the dataset.")
    st.stop()

# Prepare data for Prophet
df_train = data[['Date', 'Close']].copy()

# Convert data types safely
df_train['Date'] = pd.to_datetime(df_train['Date'], errors='coerce')

if df_train['Close'].dtype != 'float64' and df_train['Close'].dtype != 'int64':
    df_train['Close'] = pd.to_numeric(df_train['Close'], errors='coerce')

# Drop NaN values
df_train.dropna(inplace=True)

# Rename columns for Prophet
df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

# Train the Prophet model
if df_train.empty:
    st.error("Error: No valid data available after preprocessing.")
    st.stop()

m = Prophet()
m.fit(df_train)

# Predict future values
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Plot forecasted data
st.subheader('Forecast data')
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
