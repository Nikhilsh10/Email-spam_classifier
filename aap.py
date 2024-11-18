import streamlit as st
import pickle

# Load the pre-trained model and vectorizer
model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Streamlit app title
st.title("Email Spam Classification Application")
st.write("This is a Machine Learning application to classify emails as Spam or Ham.")

# Subheader for classification
st.subheader("Classification")

# User input area
user_input = st.text_area("Enter your email")


# Classify button logic
if st.button("Classify"):
    # Transform the input data using the vectorizer
    data = [user_input]  # Wrap input string in a list
    vect = cv.transform(data).toarray()
    
    # Make the prediction
    my_prediction = model.predict(vect)
    
    # Display the result based on the prediction
    if my_prediction[0] == 1:
        st.write("This email is classified as Spam.")
    else:
        st.write("This email is not spam.")