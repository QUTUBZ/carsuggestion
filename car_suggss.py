
import streamlit as st
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle

# Load the uploaded dataset
uploaded_file = 'cardemo.csv'
df = pd.read_csv(uploaded_file)

# Fill missing values if necessary
df['fuel'].fillna(df['fuel'].mode()[0], inplace=True)
df['seller_type'].fillna(df['seller_type'].mode()[0], inplace=True)
df['transmission'].fillna(df['transmission'].mode()[0], inplace=True)
df['owner'].fillna(df['owner'].mode()[0], inplace=True)
df['seats'].fillna(df['seats'].mode()[0], inplace=True)
df['engine'].fillna(df['engine'].mode()[0], inplace=True)
df['city'].fillna(df['city'].mode()[0], inplace=True)

# Preprocess categorical variables
df['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
df['fuel'].replace(['Petrol', 'Diesel', 'CNG', 'Electric', 'Hybrid'], [1, 2, 3, 4, 5], inplace=True)
df['seller_type'].replace(['Individual', 'Dealer'], [1, 2], inplace=True)
df['owner'].replace(['First Owner', 'Second Owner'], [1, 2], inplace=True)

# Define mapping dictionaries
transmission_map = {1: 'Manual', 2: 'Automatic'}
fuel_map = {1: 'Petrol', 2: 'Diesel', 3: 'CNG', 4: 'Electric', 5: 'Hybrid'}
seller_map = {1: 'Individual', 2: 'Dealer'}
owner_map = {1: 'First Owner', 2: 'Second Owner'}

# Encode city
cities = ['Ahmedabad', 'Jaipur', 'Surat', 'Mumbai', 'Hyderabad', 'Bangalore', 'Chennai', 'Pune', 'Kolkata', 'Delhi']
city_map = {city: idx for idx, city in enumerate(cities)}

# Add an encoded city column for model input
df['city_encoded'] = df['city'].map(city_map)

# Features and target
X = df[['year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'seats', 'city_encoded']]
y = df['carname']  # Target variable (car names)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a model (using KNN as an example)
model = NearestNeighbors(n_neighbors=5)
model.fit(X_scaled)

# Save the trained model and scaler using pickle
with open('car_suggestion_model.pkl', 'wb') as model_file:
    pickle.dump((model, scaler), model_file)

# Load the trained model and scaler
with open('car_suggestion_model.pkl', 'rb') as model_file:
    model, scaler = pickle.load(model_file)

# Streamlit UI
st.title('Car Suggestion System')

# Input fields for user preferences
transmission = st.selectbox('Select transmission type', ['Manual', 'Automatic'])
mileage_range = st.selectbox('Select mileage preference (kmpl)', ['10-15', '15-20', '20-25', '25-30', '30-35', '35-40'])
km_driven_range = st.selectbox('Select km driven preference (in km)', ['20000-40000', '40000-60000', '60000-80000', '80000-100000', '100000-130000'])
seller_type = st.selectbox('Select seller type', ['Individual', 'Dealer'])
owner = st.selectbox('Select owner type', ['First Owner', 'Second Owner'])
seats_options = list(sorted(df['seats'].unique()))
seats = st.selectbox('Select number of seats', seats_options)
year = st.selectbox('Select car year', sorted(df['year'].unique()))
fuel = st.selectbox('Select fuel type', ['Petrol', 'Diesel', 'CNG', 'Electric', 'Hybrid'])
city = st.selectbox('Select city', cities)

# Add checkbox and selectbox for brand selection
filter_by_brand = st.checkbox('Filter by specific brand')
if filter_by_brand:
    brand = st.selectbox('Select brand', ['MARUTI', 'HYUNDAI', 'TATA', 'MAHINDRA', 'HONDA', 'TOYOTA', 'FORD', 'VOLKSWAGEN', 'KIA', 'BMW', 'MERCEDES-BENZ', 'AUDI', 'JEEP', 'NISSAN', 'SKODA', 'VOLVO', 'LAND ROVER', 'MG', 'ISUZU', 'FIAT'])

# Priority selection
st.write('### Select your priorities (Drag to reorder)')
priorities = st.multiselect('Priorities', ['Mileage', 'Year', 'KM Driven', 'Fuel', 'City'], ['Mileage', 'Year', 'KM Driven', 'Fuel', 'City'])
# priorities_order = st.experimental_data_editor(priorities)

# Helper function to convert range to mid value
def get_mid_value(range_str):
    low, high = map(float, range_str.split('-'))
    return (low + high) // 2

# Helper function to map city codes to city names
def map_city_code_to_name(city_code):
    reverse_city_map = {v: k for k, v in city_map.items()}
    return reverse_city_map.get(city_code, 'Unknown')

# Predict function based on user input and priorities
def suggest_car(year, km_driven_range, fuel, seller_type, transmission, owner, mileage_range, seats, city, priorities, brand=None):
    # Transform user input into a format expected by the model
    transmission_code = 1 if transmission == 'Manual' else 2
    owner_code = ['First Owner', 'Second Owner'].index(owner) + 1
    fuel_code = ['Petrol', 'Diesel', 'CNG', 'Electric', 'Hybrid'].index(fuel) + 1
    seller_type_code = ['Individual', 'Dealer'].index(seller_type) + 1
    city_code = city_map[city]
    km_driven = get_mid_value(km_driven_range)
    mileage = get_mid_value(mileage_range)
    
    user_input = [year, km_driven, fuel_code, seller_type_code, transmission_code, owner_code, mileage, 0, seats, city_code]
    
    # Scale the user input
    user_input_scaled = scaler.transform([user_input])
    
    # Apply weights based on priorities
    weights = [1] * len(user_input)
    for idx, priority in enumerate(priorities):
        if priority == 'Year':
            weights[0] = (5 - idx)
        elif priority == 'KM Driven':
            weights[1] = (5 - idx)
        elif priority == 'Mileage':
            weights[6] = (5 - idx)
        elif priority == 'Fuel':
            weights[2] = (5 - idx)
        elif priority == 'City':
            weights[9] = (5 - idx)
    
    # Predict using the model
    distances, indices = model.kneighbors(user_input_scaled * weights)
    suggested_cars = df.iloc[indices[0]].reset_index(drop=True)
    
    # Filter by brand if specified
    if brand:
        suggested_cars = suggested_cars[suggested_cars['carname'].str.contains(brand, case=False)]
    
    # Map city codes back to city names for display
    suggested_cars['city'] = suggested_cars['city_encoded'].map(map_city_code_to_name)
    
    # Drop the city_encoded column from the display
    suggested_cars = suggested_cars.drop(columns=['city_encoded'])
    
    return suggested_cars

# Button to trigger suggestion
if st.button('Get Car Suggestions'):
    suggestions = suggest_car(year, km_driven_range, fuel, seller_type, transmission, owner, mileage_range, seats, city, priorities, brand if filter_by_brand else None)
    
    # Replace numeric codes with categorical values
    suggestions['transmission'] = suggestions['transmission'].map(transmission_map)
    suggestions['fuel'] = suggestions['fuel'].map(fuel_map)
    suggestions['seller_type'] = suggestions['seller_type'].map(seller_map)
    suggestions['owner'] = suggestions['owner'].map(owner_map)
    
    st.write('### Suggested Car Models')
    st.dataframe(suggestions)

