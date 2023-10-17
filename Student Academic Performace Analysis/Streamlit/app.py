import streamlit as st
import pickle

# Load the pickled model
with open('model.pkl', 'rb') as file:
    model= pickle.load(file)
st.title("Student Performance Predictor")
health = st.slider("Select the health metric", 1,5, key='health')
age = st.number_input("Enter your age", step=1, key='age')
famrel = st.slider("Select the family relationship", 1,5, key='famrel')
absences = st.number_input("Enter the number of absences", step=1, key='absences')
g2 = st.number_input("Enter the second period grade", step=1, min_value=0, max_value=20, key='g2')

if st.button("Predict"):
    prediction = model.predict([[g2, absences, age, famrel, health]])
    st.subheader(f"Your grade 3 is most likely to be {prediction[0]}")
