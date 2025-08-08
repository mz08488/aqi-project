# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# CITIES = ["Karachi", "Lahore", "Islamabad"]

# st.title("Air Quality Forecast Dashboard 🇵🇰")

# city = st.selectbox("Select City", CITIES)
# file = f"data/forecast/forecast_features_{city.lower()}.csv"

# if os.path.exists(file):
#     df = pd.read_csv(file)
#     st.subheader(f"Latest Forecast Data: {city}")
#     st.dataframe(df.tail(10))

#     st.subheader("AQI Trend Over Time")
#     fig, ax = plt.subplots()
#     df.set_index("date")["us_aqi"].plot(ax=ax)
#     ax.set_ylabel("AQI")
#     ax.set_xlabel("Date")
#     ax.set_title(f"AQI Trend in {city}")
#     st.pyplot(fig)

#     st.subheader("Pollutants: PM10 and PM2.5")
#     fig2, ax2 = plt.subplots()
#     df.set_index("date")[["pm10", "pm2_5"]].plot(ax=ax2)
#     ax2.set_ylabel("µg/m³")
#     st.pyplot(fig2)
# else:
#     st.warning("No forecast data found. Please run forecast_and_predict.py first.")
import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Page setup
st.set_page_config(page_title="Air Quality Forecast", layout="wide")
st.title("🌍 AQI Forecast Dashboard")

# Load data
DATA_PATH = "data/predicted_aqi_72hr.csv"
if not os.path.exists(DATA_PATH):
    st.error("Prediction data not found. Please ensure the CSV exists at: `data/predicted_aqi_72hr.csv`")
    st.stop()

df = pd.read_csv(DATA_PATH)
df["time"] = pd.to_datetime(df["time"])

# Only use relevant columns
df = df[["time", "predicted_us_aqi", "city"]]

# Mapping city code to name
city_map = {0: "Karachi", 1: "Islamabad", 2: "Lahore"}
df["city_name"] = df["city"].map(city_map)

# Sidebar filters
st.sidebar.header("🔎 Filters")
city_selected = st.sidebar.selectbox("Select a City", df["city_name"].unique(), index=0)

# Filter data
filtered_df = df[df["city_name"] == city_selected].copy()

# Summary box
st.subheader(f"🧭 AQI Forecast for {city_selected} (Next 72 Hours)")

latest = filtered_df.iloc[-1]
st.metric(label="Latest Predicted AQI", value=f"{latest['predicted_us_aqi']:.1f}")

# Plot forecast trend
fig = px.line(
    filtered_df,
    x="time",
    y="predicted_us_aqi",
    title=f"AQI Forecast Trend - {city_selected}",
    markers=True,
    color_discrete_sequence=["#636EFA"]
)
fig.update_layout(xaxis_title="Time", yaxis_title="Predicted AQI", height=400)
st.plotly_chart(fig, use_container_width=True)

# Display raw forecast table
st.subheader("📋 Forecast Data")
st.dataframe(filtered_df[["time", "predicted_us_aqi"]].round(2), use_container_width=True)

# Download option
csv_download = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Data as CSV", data=csv_download, file_name=f"aqi_forecast_{city_selected.lower()}.csv", mime="text/csv")