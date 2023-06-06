import streamlit as st



st.markdown("<h1 style='text-align: center; color: white;'>Text → SQL</h1>", unsafe_allow_html=True)
# description
st.markdown("<h2 style='text-align: left; color: white;'>Convert your sentence into an SQL Query 🪄</h2>", unsafe_allow_html=True)

# input box for text
text = st.text_area("Enter your sentence here")

if st.button('Generate'):
    st.write('Why hello there')

