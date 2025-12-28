import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the trained model
@st.cache_resource
def load_model():
    data = joblib.load('house_price_model_rf.pkl')
    return data['model'], data['columns']

model, model_columns = load_model()

# 2. Load locations
@st.cache_data
def get_locations():
    df = pd.read_csv('cleaned_train.csv')
    return sorted(df['location'].unique().tolist())

locations = get_locations()

# --- APP LAYOUT ---
st.title("🏡 Bangalore House Price Predictor")
st.write("Enter the details of the house to get an estimated price.")

# Input Form
col1, col2 = st.columns(2)
with col1:
    location = st.selectbox("Location", locations)
    bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2)
    bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

with col2:
    balcony = st.number_input("Balcony", min_value=0, max_value=5, value=1)
    sqft = st.number_input("Total Sqft", min_value=300, max_value=10000, value=1000)

# Trigger Button
if st.button("Estimate Price"):
    # Create input data
    input_data = pd.DataFrame({
        'bhk': [bhk],
        'bath': [bath],
        'balcony': [balcony],
        'total_sqft_numeric': [sqft]
    })
    
    # Handle One-Hot Encoding
    for col in model_columns:
        if col.startswith('location_'):
            input_data[col] = 0
            
    loc_col = f'location_{location}'
    if loc_col in input_data.columns:
        input_data[loc_col] = 1
        
    # Align columns
    input_data = input_data.reindex(columns=model_columns, fill_value=0)
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    st.success(f"💰 Estimated Price: ₹ {prediction:.2f} Lakhs")
    st.info(f"Price per Sqft: ₹ {(prediction * 100000) / sqft:.0f}")

st.caption("Built with Streamlit & Random Forest")