import streamlit as st
import pandas as pd
from joblib import load

# -----------------------------
#       PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Airline Ticket Price Prediction",
    page_icon="✈️",
    layout="wide"
)

# -----------------------------
# LOAD FILES
# -----------------------------
model = load("model.joblib")
columns = load("columns.joblib")
encoders = load("encoders.joblib")

# Excel dataset
df = pd.read_excel("airlines.xlsx")

# -----------------------------
# HEADER
# -----------------------------
st.title("✈️ Airline Ticket Price Prediction Dashboard")
st.markdown(
    "Predict airline ticket prices using Machine Learning and explore airline insights."
)

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🛫 Flight Details")

airline = st.sidebar.selectbox(
    "Airline",
    sorted(df["Airline"].unique())
)

source = st.sidebar.selectbox(
    "Source",
    sorted(df["Source"].unique())
)

destination = st.sidebar.selectbox(
    "Destination",
    sorted(df["Destination"].unique())
)

cabin = st.sidebar.selectbox(
    "Cabin Class",
    sorted(df["Cabin Class"].unique())
)

# -----------------------------
# KPI SECTION
# -----------------------------
st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Flights",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Total Airlines",
        df["Airline"].nunique()
    )

with col3:
    st.metric(
        "Average Price",
        f"₹ {round(df['Ticket Price'].mean(),0):,.0f}"
    )

with col4:
    st.metric(
        "Maximum Price",
        f"₹ {df['Ticket Price'].max():,.0f}"
    )

st.divider()

# -----------------------------
# CHARTS
# -----------------------------
st.subheader("📈 Airline Analytics")

c1, c2 = st.columns(2)

with c1:
    st.write("Flights by Airline")
    st.bar_chart(df["Airline"].value_counts())

with c2:
    st.write("Cabin Class Distribution")
    st.bar_chart(df["Cabin Class"].value_counts())

c3, c4 = st.columns(2)

with c3:
    st.write("Source Distribution")
    st.bar_chart(df["Source"].value_counts())

with c4:
    st.write("Destination Distribution")
    st.bar_chart(df["Destination"].value_counts())

st.divider()

# -----------------------------
# PRICE ANALYSIS
# -----------------------------
st.subheader("💰 Average Ticket Price By Airline")

avg_price = (
    df.groupby("Airline")["Ticket Price"]
    .mean()
    .sort_values(ascending=False)
)

st.bar_chart(avg_price)

st.divider()

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.subheader("🎯 Predict Ticket Price")

input_df = pd.DataFrame({
    "Airline": [airline],
    "Source": [source],
    "Destination": [destination],
    "Cabin Class": [cabin]
})

# Encode input
for col in input_df.columns:
    input_df[col] = encoders[col].transform(input_df[col])

input_df = input_df.reindex(
    columns=columns,
    fill_value=0
)

if st.button("Predict Ticket Price 🚀", use_container_width=True):

    prediction = model.predict(input_df)

    st.success(
        f"💸 Predicted Ticket Price : ₹ {round(prediction[0]):,}"
    )

st.divider()

# -----------------------------
# INPUT SUMMARY
# -----------------------------
st.subheader("📋 Input Summary")

summary = pd.DataFrame({
    "Airline": [airline],
    "Source": [source],
    "Destination": [destination],
    "Cabin Class": [cabin]
})

st.dataframe(summary, use_container_width=True)

# -----------------------------
# TOP AIRLINES
# -----------------------------
st.subheader("🏆 Top Expensive Airlines")

top_airlines = (
    df.groupby("Airline")["Ticket Price"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

st.dataframe(top_airlines, use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "<center>✈️ Airline Ticket Price Prediction using Machine Learning</center>",
    unsafe_allow_html=True
)