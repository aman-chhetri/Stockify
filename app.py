import streamlit as st
from datetime import date

# Importing yfinance library to fetch the stock data 
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Set the start and end date of the stock data to be downloaded  
START ="2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stockify 💹: Stock Prediction App')
st.text('Welcome to Stockify! Navigate the Future of Finance.')

# Set the stock ticker symbols 
stocks = ('AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA', 'META')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Set the number of years for prediction 
n_years = st.slider('Years of prediction:', 1, 5)
period = n_years * 365

# Load the stock data using yfinance library 
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

# Load the data 
data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...Done!")

# Display the raw data 
st.subheader('Raw data')
st.write(data.tail())

# Plot raw data on graph using plotly library 
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text='Time Series Data',xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting using Prophet 
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Create a Prophet model 
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Plot forecasted data on graph using plotly library
st.subheader('Forecast data')
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
