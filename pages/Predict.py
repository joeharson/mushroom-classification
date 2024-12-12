import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the trained model and LabelEncoder
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Load the dataset for feature options
df = pd.read_csv('mushrooms.csv')

# All features used in training
all_features = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
    'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
    'spore-print-color', 'population', 'habitat'
]

# Selected features for classification
selected_features = [ 'bruises','gill-size','gill-spacing', 'gill-color','stalk-surface-below-ring','veil-color', 'ring-type','population', 'habitat', ]

# Streamlit Interface
st.title('Mushroom Classification')

st.write("Provide values for the following features:")

# Create input fields for selected features
user_inputs = {}
for feature in selected_features:
    user_inputs[feature] = st.selectbox(feature, df[feature].unique())

# Create a complete input data array with default values
input_data = []
for feature in all_features:
    if feature in selected_features:
        if user_inputs[feature] in label_encoder.classes_:
            input_data.append(label_encoder.transform([user_inputs[feature]])[0])
        else:
            # Handle unseen labels
            input_data.append(0)  # Default value for unseen labels
    else:
        # Use default values for other features
        input_data.append(0)  # Default value

# Convert the inputs into a DataFrame for prediction
input_df = pd.DataFrame([input_data], columns=all_features)

# Prediction button
if st.button('Classify'):
    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0]  # Get probabilities for both classes

    # Extract probabilities for edible and poisonous
    edible_prob = prediction_prob[0]  # Probability for edible (class 0)
    poisonous_prob = prediction_prob[1]  # Probability for poisonous (class 1)

    # Display probabilities and result
    if prediction == 1:  # Poisonous
        st.error(f"Prediction: Poisonous (Probability: {poisonous_prob:.2%})")
        st.warning("Warning! This mushroom is highly poisonous. Avoid consumption!")
    elif prediction == 0:  # Edible
        st.success(f"Prediction: Edible (Probability: {edible_prob:.2%})")
        st.info("This mushroom is safe to eat. Enjoy!")

    # Show detailed probabilities
    st.write("Prediction Probabilities:")
    st.write(f"Edible: {edible_prob:.2%}")
    st.write(f"Poisonous: {poisonous_prob:.2%}")
