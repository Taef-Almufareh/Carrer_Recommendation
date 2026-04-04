import streamlit as st
import pandas as pd

st.title("My App")

df = pd.read_excel("career_recommendation_sample_data.xlsx")
st.dataframe(df)
