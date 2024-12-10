import streamlit as st

# Title and Header
st.title("Mushroom Classification Project")
st.header("Overview of the Mushroom Classification Process")

# Introduction
st.subheader("Introduction")
st.write("""
Mushrooms are an essential part of the ecosystem, but identifying whether a mushroom is edible or poisonous can be tricky. 
This project aims to classify mushrooms based on their features using machine learning. A Random Forest Classifier has been trained on the dataset to predict whether a mushroom is edible or poisonous.
""")

# Dataset Description
st.subheader("Dataset Details")
st.write("""
The dataset contains information on various physical features of mushrooms, including:
- Cap Shape
- Cap Surface
- Cap Color
- Odor
- Gill Color
- Habitat, and more.

The target variable is `class`, which categorizes mushrooms into:
- **Edible (e)**
- **Poisonous (p)**.
""")

# Data Statistics
st.subheader("Dataset at a Glance")
st.write("""
The dataset consists of **8124 rows** and **23 features**. All features are categorical, making it ideal for label encoding.
Below is a brief description of the dataset:
""")

st.code("""
- Number of Samples: 8124
- Features: 22 (e.g., cap shape, odor, habitat)
- Target Variable: class (edible/poisonous)
- Data Type: Categorical
""", language="markdown")

# Process Description
st.subheader("Process Overview")
st.write("""
The process for mushroom classification involves:
1. **Data Preprocessing**: Encoding categorical features using LabelEncoder.
2. **Splitting the Dataset**: Dividing the dataset into training and testing sets.
3. **Model Training**: Training a Random Forest Classifier on the training data.
4. **Prediction**: Predicting whether a mushroom is edible or poisonous based on user inputs.
5. **Evaluation**: Using metrics like accuracy, confusion matrix, and ROC curve for performance assessment.
""")

# Insights and Motivation
st.subheader("Insights and Motivation")
st.write("""
This classification model can assist individuals in making informed decisions about mushrooms found in nature.
It demonstrates the power of machine learning in handling categorical data efficiently and provides a practical application of Random Forest models.
""")

# Footer
st.write("💡 **Note**: Always consult an expert when foraging for mushrooms in the wild.")
