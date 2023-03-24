import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Set up the app title and header
st.title("Iris Flower Species Prediction")
st.header("Enter the details of the flower to predict its species")

# Load the iris dataset
iris = datasets.load_iris()

# Assign the data and target variables
X = iris.data
Y = iris.target

# Create a Random Forest Classifier
rfc = RandomForestClassifier()

# Fit the model to the data
rfc.fit(X, Y)

# Add input fields for the flower details
sepal_length = st.slider("Sepal length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Define the prediction button
if st.button("Predict"):
    # Make a prediction using the Random Forest Classifier
    prediction = rfc.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    # Get the predicted species name
    species = iris.target_names[prediction[0]]

    # Display the predicted species
    st.write(f"The predicted species is {species}")
