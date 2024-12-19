import logging
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# Configure logging
logging.basicConfig(
    filename='mushroom_classification.log',  # Log file name
    level=logging.INFO,  # Set log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

# Load the trained model and LabelEncoder
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

try:
    with open('label_encoder.pkl', 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)
    logging.info("LabelEncoder loaded successfully.")
except Exception as e:
    logging.error(f"Error loading LabelEncoder: {e}")

# Load the dataset for feature options
try:
    df = pd.read_csv('mushrooms.csv')
    logging.info("Dataset loaded successfully.")
except Exception as e:
    logging.error(f"Error loading dataset: {e}")

# All features used in training
all_features = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
    'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
    'spore-print-color', 'population', 'habitat'
]

# Selected features for classification
selected_features = [
    'bruises', 'gill-size', 'gill-spacing', 'gill-color',
    'stalk-surface-below-ring', 'veil-color', 'ring-type', 'population', 'habitat',
]

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
            # If the input is in the encoder's classes, encode it
            input_data.append(label_encoder.transform([user_inputs[feature]])[0])
        else:
            # Handle unseen labels by appending a default value (0)
            logging.warning(f"Unseen label encountered: {user_inputs[feature]} for feature {feature}")
            st.warning(f"Unseen label '{user_inputs[feature]}' encountered for {feature}. Defaulting to '0'.")
            input_data.append(0)  # Default value for unseen labels
    else:
        # Use default values for other features
        input_data.append(0)  # Default value

# Convert the inputs into a DataFrame for prediction
input_df = pd.DataFrame([input_data], columns=all_features)

# Prediction button
if st.button('Classify'):
    try:
        prediction = model.predict(input_df)[0]
        prediction_prob = model.predict_proba(input_df)[0]  # Get probabilities for both classes
        logging.info("Prediction made successfully.")
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error("An error occurred during prediction. Please check the logs for more details.")
        prediction, prediction_prob = None, None

    if prediction is not None:
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
