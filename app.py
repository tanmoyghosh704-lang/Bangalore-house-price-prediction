import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('RidgeModel.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure RidgeModel.pkl is in the same directory.")
        return None

# Load locations
@st.cache_data
def load_locations():
    try:
        df = pd.read_csv('cleaned_data.csv')
        locations = sorted(df['location'].unique())
        return locations
    except FileNotFoundError:
        # Default locations if file not found
        return ['1st Block Jayanagar', '1st Phase JP Nagar', '2nd Phase Judicial Layout', 
                '2nd Stage Nagarbhavi', '5th Block Hbr Layout', '5th Phase JP Nagar',
                '6th Phase JP Nagar', '7th Phase JP Nagar', '8th Phase JP Nagar',
                'AECS Layout', 'Abbigere', 'Akshaya Nagar', 'other']

# Title and description
st.title("🏠 Bangalore House Price Predictor")
st.markdown("### Predict house prices in Bangalore based on location, size, and amenities")

# Load model and locations
model = load_model()
locations = load_locations()

if model is not None:
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Property Details")
        
        # Location selection
        location = st.selectbox(
            "Select Location",
            options=locations,
            help="Choose the location of the property"
        )
        
        # BHK selection
        bhk = st.number_input(
            "Number of BHK",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Number of bedrooms"
        )
        
    with col2:
        st.subheader("Property Specifications")
        
        # Total square feet
        total_sqft = st.number_input(
            "Total Square Feet",
            min_value=300.0,
            max_value=10000.0,
            value=1000.0,
            step=50.0,
            help="Total area in square feet"
        )
        
        # Number of bathrooms
        bath = st.number_input(
            "Number of Bathrooms",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=1.0,
            help="Number of bathrooms"
        )
    
    # Add some spacing
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'location': [location],
                'total_sqft': [total_sqft],
                'bath': [bath],
                'bhk': [bhk]
            })
            
            # Make prediction
            predicted_price = model.predict(input_data)[0]
            
            # Display results
            st.success("### Prediction Results")
            
            # Create three columns for metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="Predicted Price",
                    value=f"₹{predicted_price:.2f} Lakhs"
                )
            
            with metric_col2:
                price_per_sqft = (predicted_price * 100000) / total_sqft
                st.metric(
                    label="Price per Sq Ft",
                    value=f"₹{price_per_sqft:.2f}"
                )
            
            with metric_col3:
                st.metric(
                    label="Total Price",
                    value=f"₹{predicted_price * 100000:,.0f}"
                )
            
            # Additional insights
            st.info(f"""
            **Property Summary:**
            - **Location:** {location}
            - **Configuration:** {bhk} BHK
            - **Area:** {total_sqft} sq ft
            - **Bathrooms:** {int(bath)}
            - **Estimated Price:** ₹{predicted_price:.2f} Lakhs (₹{predicted_price * 100000:,.0f})
            """)
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    
    # Add footer with information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>This prediction is based on a Ridge Regression model trained on Bangalore housing data.</p>
        <p>Prices are in Indian Rupees (Lakhs). 1 Lakh = 100,000 INR</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("Please ensure the model file (RidgeModel.pkl) is available in the app directory.")

# Sidebar with additional information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app predicts house prices in Bangalore using machine learning.
    
    **Features:**
    - Location-based pricing
    - BHK configuration
    - Total area consideration
    - Bathroom count
    
    **Model:**
    - Algorithm: Ridge Regression
    - Preprocessing: OneHotEncoding + StandardScaler
    
    **Tips for accurate predictions:**
    - Ensure the location is from Bangalore
    - Use realistic square footage values
    - BHK and bathroom count should be proportional to area
    """)
    
    st.markdown("---")
    st.markdown("**Developer Note:**")
    st.caption("Built with Streamlit and scikit-learn")