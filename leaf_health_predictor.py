import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

# Let's generate a dataset with 500 samples
np.random.seed(42)  # This has been set for reproducibility

def generate_data(n_samples):
    leaf_colors = np.random.choice([0, 1, 2], n_samples)  # 0: Green, 1: Yellow, 2: Brown
    leaf_lengths = np.random.uniform(5.0, 20.0, n_samples)  # Leaf Length in cm
    leaf_widths = np.random.uniform(2.0, 10.0, n_samples)  # Leaf Width in cm
    leaf_spots = np.random.choice([0, 1], n_samples)  # 0: No, 1: Yes
    health_status = np.random.choice([0, 1, 2], n_samples)  # 0: Healthy, 1: Needs Water, 2: Diseased

    data = pd.DataFrame({
        'Leaf Color': leaf_colors,
        'Leaf Length': leaf_lengths,
        'Leaf Width': leaf_widths,
        'Leaf Spots': leaf_spots,
        'Health': health_status
    })

    return data

data = generate_data(500)

# Split the data
X = data[['Leaf Color', 'Leaf Length', 'Leaf Width', 'Leaf Spots']]
y = data['Health']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Calculate feature importances
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Working on the streamlit UI

# About Page
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Predictor", "About"])

if page == "About":
    st.title("About the Leaf Health Predictor")
    st.write("""
    This **Leaf Health Predictor** will help users assess the health of plants based on features such as leaf color, size, and the presence of spots. 
    It was built using a Decision Tree classifier to provide a basic prediction of a plant’s health status: Healthy, Needs Water, or Diseased.
    This project provides a foundational understanding of deploying machine learning models in real-world scenarios.
    """)
    st.write("""
    **Explanation of features**:
    
    Leaf color Values:
    - **Green (0)**: Healthy or well-nourished leaves.
    - **Yellow (1)**: Possible nutrient deficiency or stress.
    - **Brown (2)**: Indication of disease, dehydration, or aging.
    
    Leaf lenght and width:
    - represents leaf measurements in cm
    - they could be very small or underdeveloped leaves or fully developed or large leaves.
    
    Leaf Health Values:
    - **Yes (1)**: The leaf has spots, indicating possible disease or pest damage.
    - **No (0)**: The leaf is free of spots, suggesting that it is healthy or not affected by visible diseases.
    """)
else:
    st.title("Leaf Health Predictor")
    st.write("Dataset Overview")
    st.dataframe(data.head())

    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Inputs for prediction
    leaf_color = st.selectbox("Select Leaf Color", ["Green", "Yellow", "Brown"])
    leaf_length = st.number_input("Enter Leaf Length (cm)", min_value=0.0, format="%.2f")
    leaf_width = st.number_input("Enter Leaf Width (cm)", min_value=0.0, format="%.2f")
    leaf_spots = st.selectbox("Does the leaf have spots?", ["Yes", "No"])
    spots_value = 1 if leaf_spots == "Yes" else 0

    # Create input data for prediction
    input_data = pd.DataFrame({
        'Leaf Color': [leaf_color],
        'Leaf Length': [leaf_length],
        'Leaf Width': [leaf_width],
        'Leaf Spots': [spots_value]
    })
    input_data['Leaf Color'] = input_data['Leaf Color'].map({'Green': 0, 'Yellow': 1, 'Brown': 2})

    # Prediction
    predicted_health = model.predict(input_data)[0]
    health_map = {0: "Healthy", 1: "Needs Water", 2: "Diseased"}
    st.write(f"Predicted Health: {health_map[predicted_health]}")

    # Feature importance plot
    fig = px.bar(importance_df, x='Feature', y='Importance',
                 title='Feature Importances',
                 labels={'Importance': 'Importance Score', 'Feature': 'Features'},
                 height=400, width=600)

    st.plotly_chart(fig)
