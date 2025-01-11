import streamlit as st
import pickle 

model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))

  # Example image URL
st.title("📧 SMS Spam Detection")
st.subheader("Detect whether an email is spam or not spam using Machine Learning.")
st.write("This app uses a Machine Learning model to analyze email content and classify it as **Spam** or **Not Spam**.")

user_input=st.text_area("✍️Enter an Email to classify",height=100)

if st.button("Classify"):
    if user_input:
        data=[user_input]
        vectorized_data = cv.transform(data).toarray()
        result = model.predict(vectorized_data)
        if result[0]==0:
             st.success("✅ The Email is **not spam**.")
             
        else:
             st.warning("🚨 The Email is **spam**!")
      else:
        st.warning("⚠️Please type Email to classify")
