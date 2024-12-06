import requests
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

# Hardcoded API Key
API_KEY = "02061ee61d38f43a26a134f24e9041d6"  

# Set Streamlit page config
st.set_page_config(
    page_title="Weather Prediction Model",
    page_icon="🌤️",
    layout="wide"
)


def set_bg_color():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f3f4f6;
        }
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
        }
        .subtitle {
            font-size: 20px;
            color: #34495e;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Fetch Weather Data from OpenWeatherMap API
def fetch_weather_data(city, days):
    base_url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {"q": city, "cnt": days * 8, "units": "metric", "appid": API_KEY}
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200:
        st.error(f"API Error: {response.json().get('message', 'Unknown error')}")
        return None
    
    data = response.json()
    weather_list = data["list"]
    weather_data = [
        {
            "date_time": item["dt_txt"],
            "temperature": item["main"]["temp"],
            "humidity": item["main"]["humidity"],
            "wind_speed": item["wind"]["speed"],
            "pressure": item["main"]["pressure"]
        }
        for item in weather_list
    ]
    return pd.DataFrame(weather_data)


# Train a Weather Prediction Model
def train_model(data):
    # Feature Engineering
    data["hour"] = pd.to_datetime(data["date_time"]).dt.hour
    data["day"] = pd.to_datetime(data["date_time"]).dt.day
    features = ["hour", "day", "humidity", "wind_speed", "pressure"]
    target = "temperature"
    
    X = data[features]
    y = data[target]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict Temperatures
    data["predicted_temperature"] = model.predict(X[features])
    return data

# Visualize Trends
def plot_trends(data):
    fig = px.line(
        data, 
        x="date_time", 
        y=["temperature", "predicted_temperature"], 
        title="Temperature Trends: Actual vs Predicted",
        labels={"value": "Temperature (°C)", "variable": "Legend"},
        template="plotly_white"
    )
    return fig

# Streamlit Interactive App
def main():
    # Set background color
    set_bg_color()

    st.markdown("<h1 class='title'>Weather Prediction Platform 🌤️</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Analyze and predict weather trends with ease</p>", unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.header("Settings")
city = st.sidebar.text_input("City", placeholder="Enter a city name")
days = st.sidebar.slider("Forecast Days", 1, 5, 3)

if st.sidebar.button("Fetch Data 🌍"):
    if city:  # Ensure a city name is provided
        with st.spinner("Fetching weather data..."):
            data = fetch_weather_data(city, days)
            if data is not None:
                with st.expander(f"Weather Data for {city.capitalize()}"):
                    st.dataframe(data)
                
                data_with_predictions = train_model(data)
                st.markdown("### Temperature Trends")
                fig = plot_trends(data_with_predictions)
                st.plotly_chart(fig)
    else:
        st.error("Please enter a city name!")


if __name__ == "__main__":
    main()
