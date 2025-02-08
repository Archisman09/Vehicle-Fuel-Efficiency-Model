# 1. Imports
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Page Configuration
st.set_page_config(
    page_title="Car MPG Predictor",
    page_icon="🚗",
    layout="wide"
)

# 3. Model Loading Function
@st.cache_resource  # This decorator caches the model to avoid reloading
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    try:
        with open('feature_names.pkl', 'rb') as file:
            expected_columns = pickle.load(file)  # Load feature names
    except FileNotFoundError:
        expected_columns = ['cylinders', 'displacement', 'horsepower', 'weight', 
                            'acceleration', 'model_year', 'origin_Europe', 'origin_Asia']
    return model, scaler, expected_columns

# 4. Main Application Function
def main():
    st.title("🚗 Car MPG Predictor")
    st.write("Enter car specifications to predict Miles Per Gallon (MPG)")
    
    # 5. Input Layout in Two Columns
    col1, col2 = st.columns(2)
    
    # First column inputs
    with col1:
        cylinders = st.selectbox(
            "Number of Cylinders", 
            options=[3, 4, 5, 6, 8]
        )
        displacement = st.number_input(
            "Displacement (cu. inches)", 
            min_value=50.0, 
            max_value=500.0, 
            value=200.0
        )
        horsepower = st.number_input(
            "Horsepower", 
            min_value=50.0, 
            max_value=500.0, 
            value=150.0
        )
    
    # Second column inputs
    with col2:
        weight = st.number_input(
            "Weight (lbs)", 
            min_value=1500.0, 
            max_value=5500.0, 
            value=3000.0
        )
        acceleration = st.number_input(
            "Acceleration (0-60 mph)", 
            min_value=5.0, 
            max_value=25.0, 
            value=15.0
        )
        model_year = st.slider(
            "Model Year", 
            min_value=70, 
            max_value=82, 
            value=76
        )
        origin = st.selectbox(
            "Origin", 
            options=['America', 'Europe', 'Asia']
        )
    
    # 6. Prediction Button and Logic
    if st.button("Predict MPG"):
        try:
            # Load model and scaler
            model, scaler, expected_columns = load_model()
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'cylinders': [cylinders],
                'displacement': [displacement],
                'horsepower': [horsepower],
                'weight': [weight],
                'acceleration': [acceleration],
                'model_year': [model_year],
                #'origin_America': [1 if origin == 'America' else 0],
                #'origin_Europe': [1 if origin == 'Europe' else 0],
                #'origin_Asia': [1 if origin == 'Asia' else 0]
                'origin': [origin]  # Keep as categorical initially
            })
            # Convert 'origin' into dummies (drop_first=True to match training)
            input_data = pd.get_dummies(input_data, columns=['origin'], drop_first=True)

            # Ensure input columns match the trained model's features
            input_data = input_data.reindex(columns=expected_columns, fill_value=0)
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Display prediction
            st.success(f"Predicted MPG: {prediction:.2f}")
            
            # Add confidence note
            st.info("Note: This prediction is based on historical car data from 1970-1982.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
# 7. Batch Prediction Section
    st.header("Batch Prediction")
    st.write("Upload a CSV file with multiple car specifications")
    
    # Show required format
    st.write("Required CSV columns:")
    st.code("cylinders,displacement,horsepower,weight,acceleration,model_year,origin")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            # Read CSV
            batch_data = pd.read_csv(uploaded_file)
            
            # Print column names for debugging
            st.write("Columns in your file:", batch_data.columns.tolist())
            
            required_columns = ['cylinders', 'displacement', 'horsepower', 
                              'weight', 'acceleration', 'model_year', 'origin']
            
            # Clean column names
            batch_data.columns = batch_data.columns.str.strip().str.lower().str.replace(' ', '_')
            
            # Validate columns
            if not all(col in batch_data.columns for col in required_columns):
                st.error("CSV must contain all required columns: " + 
                        ", ".join(required_columns))
            else:
                # Create dummy variables for origin
                batch_data_encoded = pd.get_dummies(batch_data, columns=['origin'])
                
                # Ensure all necessary columns exist
                expected_columns = ['cylinders', 'displacement', 'horsepower', 'weight',
                                  'acceleration', 'model_year', 'origin_America',
                                  'origin_Europe', 'origin_Asia']
                
                # Add missing columns if any
                for col in ['origin_America', 'origin_Europe', 'origin_Asia']:
                    if col not in batch_data_encoded.columns:
                        batch_data_encoded[col] = 0
                
                # Select only the required columns in the correct order
                batch_features = batch_data_encoded[expected_columns]
                
                # Load model and scaler
                model, scaler = load_model()
                
                # Scale data and predict
                batch_scaled = scaler.transform(batch_features)
                batch_predictions = model.predict(batch_scaled)
                
                # Add predictions to original dataframe
                batch_data['Predicted_MPG'] = batch_predictions
                
                # Show results
                st.write("Predictions:")
                st.dataframe(batch_data)
                
                # Download button
                csv = batch_data.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="mpg_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your CSV file has these exact columns:")
            for i, col in enumerate(required_columns):
                st.write(f"{i}: {col}")

# 8. Run the application
if __name__ == '__main__':
    main()