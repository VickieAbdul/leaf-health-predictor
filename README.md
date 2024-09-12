# Leaf Health Predictor
I created this Leaf Health Predictor App to help users assess the health of plants based on features such as leaf color, size, and the presence of spots. 
It was built using a Decision Tree classifier to provide a basic prediction of a plant’s health status: Healthy, Needs Water, or Diseased. 
This serves as an introduction to building interactive machine learning applications with Streamlit.

# Relevance to Agriculture and Horticulture
The app, though simple, still reflects a real-world problem in the agriculture and horticulture sectors, where monitoring plant health is critical for crop yield and management. 
By automating the process of assessing plant health, more advanced versions of this app could be used by farmers, gardeners, and horticulturists to identify early signs of disease or water stress, ultimately improving productivity and sustainability.

# My Perspective
The app is intended as a beginner-level project, allowing me to get familiar with Streamlit and learn how to deploy machine learning models in a user-friendly interface. 
The project is focused on a practical, real-world use case, making it easy to see the connection between machine learning and its applications in various industries.

# Explanation of features
#### Leaf Color:
This indicates the plant's health. Healthy leaves are typically green, while yellow or brown leaves often indicate issues like nutrient deficiency, disease, or over/under-watering.
#### Values:
- Green (0): Healthy or well-nourished leaves.
- Yellow (1): Possible nutrient deficiency or stress.
- Brown (2): Indication of disease, dehydration, or aging.

#### Leaf Size:
Reflects the overall growth and health of the plant. In ideal conditions, leaves tend to grow to their full potential. Smaller or stunted leaves may indicate stress, malnutrition, or poor growing conditions.
#### Values:
Leaf size is represented as an integer (ranging from 1 to 10 in the model), where:
- 1 represents a very small or underdeveloped leaf.
- 10 represents a fully developed and healthy leaf.

#### Leaf Spots:
Spots on a leaf is a common sign of disease or pest damage. Spots may indicate fungal or bacterial infections, which can deteriorate the plant’s health.
#### Values:
- Yes (1): The leaf has spots, indicating possible disease or pest damage.
- No (0): The leaf is free of spots, suggesting that it is healthy or not affected by visible diseases.
