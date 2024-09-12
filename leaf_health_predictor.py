import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

# Let's create a synthetic dataset
np.random.seed(42)
n_samples = 500
leaf_colors = np.random.choice(['Green', 'Yellow', 'Brown'], size=n_samples)
leaf_sizes = np.random.randint(1, 10, size=n_samples)
leaf_spots = np.random.choice([0, 1], size=n_samples)
health_status = np.where(
    (leaf_colors == 'Green') & (leaf_sizes > 3) & (leaf_spots == 0), 'Healthy',
    np.where(
        (leaf_colors == 'Yellow') | (leaf_sizes <= 3), 'Needs Water',
        'Diseased'
    )
)

# Convert the generated data to DataFrame and rename columns
df = pd.DataFrame({
    'Leaf Color': leaf_colors,
    'Leaf Size': leaf_sizes,
    'Leaf Spots': leaf_spots,
    'Health Status': health_status
})

# Feature engineering and model creation

# Encode categorical features
df['Leaf Color'] = df['Leaf Color'].map({'Green': 0, 'Yellow': 1, 'Brown': 2})

# Split data
X = df[['Leaf Color', 'Leaf Size', 'Leaf Spots']]
y = df['Health Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app UI

# The About Page
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
    Explanation of features:
    Leaf color Values:
    - **Green (0)**: Healthy or well-nourished leaves.
    - **Yellow (1)**: Possible nutrient deficiency or stress.
    - **Brown (2)**: Indication of disease, dehydration, or aging.
    Leaf size Values (integer range from 1 - 10):
    - **1** represents a very small or underdeveloped leaf.
    - **10** represents a fully developed and healthy leaf.
    Leaf Health Values:
    - **Yes (1)**: The leaf has spots, indicating possible disease or pest damage.
    - **No (0)**: The leaf is free of spots, suggesting that it is healthy or not affected by visible diseases.
    """)
else:
    st.title("Leaf Health Predictor")
    
    st.write("Dataset Overview")
    st.dataframe(df.head())
    
    st.write(f"Model Accuracy: {accuracy:.2f}")

    # Input fields for prediction
    st.write("Use slider and radio button below to check Plant Health:")
    leaf_color = st.selectbox("Leaf Color", ['Green', 'Yellow', 'Brown'])
    leaf_size = st.slider("Leaf Size", 1, 10)
    leaf_spots = st.radio("Leaf Spots", [0, 1])

    # Prepare input for prediction
    input_data = pd.DataFrame({
        'Leaf Color': [leaf_color],
        'Leaf Size': [leaf_size],
        'Leaf Spots': [leaf_spots]
    })
    input_data['Leaf Color'] = input_data['Leaf Color'].map({'Green': 0, 'Yellow': 1, 'Brown': 2})

    predicted_health = model.predict(input_data)[0]
    st.write(f"Predicted Plant Health: {predicted_health}")

    # Get feature importances from the model
    feature_importances = model.feature_importances_
    feature_names = ['Leaf Color', 'Leaf Size', 'Leaf Spots']

    # Create a DataFrame for Plotly
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # Create the feature importances plot using Plotly
    fig = px.bar(importance_df, x='Feature', y='Importance', 
                 title='Feature Importances',
                 labels={'Importance': 'Importance Score', 'Feature': 'Features'},
                 height=400, width=600)  # Adjusting plot size

    # Display the plot
    st.plotly_chart(fig)
