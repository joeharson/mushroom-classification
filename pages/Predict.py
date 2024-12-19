import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model using pickle
with open('mushroom_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app title
st.markdown(
    """
    <h1 style='text-align: center; color: #E75480;'>Mushroom Classification App</h1>
    """,
    unsafe_allow_html=True,
)

# Brief description
st.markdown(
    """
    <p style='text-align: center;'>This app predicts whether a mushroom is edible or poisonous based on various features.</p>
    """,
    unsafe_allow_html=True,
)

# Custom CSS for centering and styling
st.markdown(
    """
    <style>
    .block-container {
        max-width: 800px;
        margin: auto;
    }
    .stButton > button {
        background-color: #E75480;
        color: white;
        border: None;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #FF69B4;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar removed; all widgets in the center
with st.form("mushroom_form"):
    st.write("### Input Mushroom Features:")
    
    # Input fields for each feature
    cap_surface = st.selectbox('Cap Surface', ['smooth', 'scaly', 'fibrous', 'grooves'])
    cap_color = st.selectbox('Cap Color', ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'])
    bruises = st.selectbox('Bruises', ['bruises', 'no bruises'])
    odor = st.selectbox('Odor', ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'])
    gill_spacing = st.selectbox('Gill Spacing', ['close', 'crowded', 'distant'])
    gill_size = st.selectbox('Gill Size', ['broad', 'narrow'])
    gill_color = st.selectbox('Gill Color', ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'])
    stalk_shape = st.selectbox('Stalk Shape', ['enlarging', 'tapering'])
    stalk_root = st.selectbox('Stalk Root', ['bulbous', 'club', 'cup', 'equal', 'rhizomorphs', 'rooted', 'missing'])
    stalk_surface_above_ring = st.selectbox('Stalk Surface Above Ring', ['fibrous', 'scaly', 'silky', 'smooth'])
    stalk_surface_below_ring = st.selectbox('Stalk Surface Below Ring', ['fibrous', 'scaly', 'silky', 'smooth'])
    stalk_color_above_ring = st.selectbox('Stalk Color Above Ring', ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
    stalk_color_below_ring = st.selectbox('Stalk Color Below Ring', ['brown', 'buff', 'cinnamon', 'gray', 'orange', 'pink', 'red', 'white', 'yellow'])
    veil_color = st.selectbox('Veil Color', ['brown', 'orange', 'white', 'yellow'])
    ring_number = st.selectbox('Ring Number', ['none', 'one', 'two'])
    ring_type = st.selectbox('Ring Type', ['cobwebby', 'evanescent', 'flaring', 'large', 'none', 'pendant', 'sheathing', 'zone'])
    spore_print_color = st.selectbox('Spore Print Color', ['black', 'brown', 'buff', 'chocolate', 'green', 'orange', 'purple', 'white', 'yellow'])
    population = st.selectbox('Population', ['abundant', 'clustered', 'numerous', 'scattered', 'several', 'solitary'])
    habitat = st.selectbox('Habitat', ['grasses', 'leaves', 'meadows', 'paths', 'urban', 'waste', 'woods'])

    # Submit button
    submitted = st.form_submit_button("Predict")

# Encode inputs manually based on your mappings
if submitted:
    label_encoders = {
    'cap-surface': {'smooth': 0, 'scaly': 1, 'fibrous': 2, 'grooves': 3},
    'cap-color': {'brown': 0, 'buff': 1, 'cinnamon': 2, 'gray': 3, 'green': 4, 'pink': 5, 'purple': 6, 'red': 7, 'white': 8, 'yellow': 9},
    'bruises': {'no bruises': 0, 'bruises': 1},
    'odor': {'almond': 0, 'anise': 1, 'creosote': 2, 'fishy': 3, 'foul': 4, 'musty': 5, 'none': 6, 'pungent': 7, 'spicy': 8},
    'gill-spacing': {'close': 0, 'crowded': 1, 'distant': 2},
    'gill-size': {'narrow': 0, 'broad': 1},
    'gill-color': {'black': 0, 'brown': 1, 'buff': 2, 'chocolate': 3, 'gray': 4, 'green': 5, 'orange': 6, 'pink': 7, 'purple': 8, 'red': 9, 'white': 10, 'yellow': 11},
    'stalk-shape': {'enlarging': 0, 'tapering': 1},
    'stalk-root': {'bulbous': 0, 'club': 1, 'cup': 2, 'equal': 3, 'rhizomorphs': 4, 'rooted': 5, 'missing': 6},
    'stalk-surface-above-ring': {'fibrous': 0, 'scaly': 1, 'silky': 2, 'smooth': 3},
    'stalk-surface-below-ring': {'fibrous': 0, 'scaly': 1, 'silky': 2, 'smooth': 3},
    'stalk-color-above-ring': {'brown': 0, 'buff': 1, 'cinnamon': 2, 'gray': 3, 'orange': 4, 'pink': 5, 'red': 6, 'white': 7, 'yellow': 8},
    'stalk-color-below-ring': {'brown': 0, 'buff': 1, 'cinnamon': 2, 'gray': 3, 'orange': 4, 'pink': 5, 'red': 6, 'white': 7, 'yellow': 8},
    'veil-color': {'brown': 0, 'orange': 1, 'white': 2, 'yellow': 3},
    'ring-number': {'none': 0, 'one': 1, 'two': 2},
    'ring-type': {'cobwebby': 0, 'evanescent': 1, 'flaring': 2, 'large': 3, 'none': 4, 'pendant': 5, 'sheathing': 6, 'zone': 7},
    'spore-print-color': {'black': 0, 'brown': 1, 'buff': 2, 'chocolate': 3, 'green': 4, 'orange': 5, 'purple': 6, 'white': 7, 'yellow': 8},
    'population': {'abundant': 0, 'clustered': 1, 'numerous': 2, 'scattered': 3, 'several': 4, 'solitary': 5},
    'habitat': {'grasses': 0, 'leaves': 1, 'meadows': 2, 'paths': 3, 'urban': 4, 'waste': 5, 'woods': 6}
}

    # Encoded features
    encoded_data = {
    'cap-surface': label_encoders['cap-surface'][cap_surface],
    'cap-color': label_encoders['cap-color'][cap_color],
    'bruises': label_encoders['bruises'][bruises],
    'odor': label_encoders['odor'][odor],
    'gill-spacing': label_encoders['gill-spacing'][gill_spacing],
    'gill-size': label_encoders['gill-size'][gill_size],
    'gill-color': label_encoders['gill-color'][gill_color],
    'stalk-shape': label_encoders['stalk-shape'][stalk_shape],
    'stalk-root': label_encoders['stalk-root'][stalk_root],
    'stalk-surface-above-ring': label_encoders['stalk-surface-above-ring'][stalk_surface_above_ring],
    'stalk-surface-below-ring': label_encoders['stalk-surface-below-ring'][stalk_surface_below_ring],
    'stalk-color-above-ring': label_encoders['stalk-color-above-ring'][stalk_color_above_ring],
    'stalk-color-below-ring': label_encoders['stalk-color-below-ring'][stalk_color_below_ring],
    'veil-color': label_encoders['veil-color'][veil_color],
    'ring-number': label_encoders['ring-number'][ring_number],
    'ring-type': label_encoders['ring-type'][ring_type],
    'spore-print-color': label_encoders['spore-print-color'][spore_print_color],
    'population': label_encoders['population'][population],
    'habitat': label_encoders['habitat'][habitat]
}
    
    encoded_input = pd.DataFrame([encoded_data])
    
    # Prediction
    prediction = model.predict(encoded_input)
    
    # Display the result
    if prediction == 1:
        st.success("The mushroom is *edible*!")
    else:
        st.error("The mushroom is *poisonous*!")
