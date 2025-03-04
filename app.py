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

stocks = ('AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 5)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        if data.empty:
            st.error("No data found for the selected stock.")
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... Done!")

if data is None or data.empty:
    st.stop()

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# **Check if required columns exist**
if 'Date' not in data.columns or 'Close' not in data.columns:
    st.error("Required columns ('Date', 'Close') are missing. Check the dataset format.")
    st.write("Columns available:", data.columns.tolist())
    st.stop()

df_train = data[['Date', 'Close']].copy()
df_train.dropna(inplace=True)

df_train['Date'] = pd.to_datetime(df_train['Date'], errors='coerce')

# **Check before conversion**
st.write("Data types before conversion:", df_train.dtypes)

if not pd.api.types.is_numeric_dtype(df_train['Close']):
    df_train['Close'] = pd.to_numeric(df_train['Close'], errors='coerce')

df_train.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

# **Final check**
if df_train.empty:
    st.error("No valid data available after preprocessing.")
    st.stop()

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
