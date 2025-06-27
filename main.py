import streamlit as st
import pandas as pd
import pickle
import os

# 🌐 Page Configuration
st.set_page_config(page_title="Beer Servings Estimation", layout="centered")
st.success("✅ Streamlit app started!")  # Debug message

# 🏷️ Title & Description
st.title("🍺 Beer Servings Estimation App")
st.write("Predict the **total litres of pure alcohol** based on beverage consumption details.")

# 🔍 Show Current Working Directory
st.caption(f"📁 Looking in: `{os.getcwd()}`")

# 📦 Load the trained model
model_path = os.path.join("model", "LR_model.pkl")

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: `{model_path}`")
    st.stop()

try:
    with open(model_path, "rb") as f:
        LR_model = pickle.load(f)
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# 🧾 Input Section
st.subheader("🔧 Enter Your Data")

country = st.selectbox("🌍 Select a country", [
    "Germany", "USA", "India", "Brazil", "Czech Republic", "Ireland", "Japan"
])

continent = st.selectbox("🗺️ Select a continent", [
    "Europe", "North America", "Asia", "South America", "Africa", "Oceania"
])

beer = st.number_input("🍺 Beer Servings", min_value=0, value=50)
spirit = st.number_input("🥃 Spirit Servings", min_value=0, value=50)
wine = st.number_input("🍷 Wine Servings", min_value=0, value=50)

# 🎯 Prediction
if st.button("Predict Total Alcohol Consumption"):
    input_df = pd.DataFrame({
        'country': [country],
        'beer_servings': [beer],
        'spirit_servings': [spirit],
        'wine_servings': [wine],
        'continent': [continent]
    })

    st.write("📋 Input Preview:")
    st.dataframe(input_df)

    try:
        prediction = LR_model.predict(input_df)
        st.success(f"🎯 Estimated Total Litres of Pure Alcohol: **{round(prediction[0], 2)} litres**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
