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

# Define manual encoding (similar to your second code)
label_encoders = {
    'bruises': {'no bruises': 0, 'bruises': 1},
    'gill-size': {'narrow': 0, 'broad': 1},
    'gill-spacing': {'close': 0, 'crowded': 1, 'distant': 2},
    'gill-color': {'black': 0, 'brown': 1, 'buff': 2, 'chocolate': 3, 'gray': 4, 'green': 5, 'orange': 6, 'pink': 7, 'purple': 8, 'red': 9, 'white': 10, 'yellow': 11},
    'stalk-surface-below-ring': {'fibrous': 0, 'scaly': 1, 'silky': 2, 'smooth': 3},
    'veil-color': {'brown': 0, 'orange': 1, 'white': 2, 'yellow': 3},
    'ring-type': {'cobwebby': 0, 'evanescent': 1, 'flaring': 2, 'large': 3, 'none': 4, 'pendant': 5, 'sheathing': 6, 'zone': 7},
    'population': {'abundant': 0, 'clustered': 1, 'numerous': 2, 'scattered': 3, 'several': 4, 'solitary': 5},
    'habitat': {'grasses': 0, 'leaves': 1, 'meadows': 2, 'paths': 3, 'urban': 4, 'waste': 5, 'woods': 6}
}

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
        if user_inputs[feature] in label_encoders[feature]:
            input_data.append(label_encoders[feature][user_inputs[feature]])
        else:
            # Handle unseen labels
            logging.warning(f"Unseen label encountered: {user_inputs[feature]} for feature {feature}")
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

        # Display probabilities and result with full words
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
