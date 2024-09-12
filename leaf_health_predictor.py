import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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
st.title("Plant Health Checker")
st.write("Dataset Overview:")
st.dataframe(df.head())

st.write(f"Model Accuracy: {accuracy:.2f}")

# Input fields for prediction
st.write("Check Plant Health:")
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

# Plot feature importance
fig, ax = plt.subplots()
importance = model.feature_importances_
features = X.columns
ax.bar(features, importance)
ax.set_xlabel('Features')
ax.set_ylabel('Importance')
st.pyplot(fig)
