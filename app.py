import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_wine

scaler_model = pickle.load(open('model_scaler.pkl','rb')) 
logistic_model = pickle.load(open('model_lr.pkl','rb'))
dtc_model = pickle.load(open('model_dtc.pkl','rb'))

#load_data
wine = load_wine()
wine_df = pd.DataFrame(wine.data,columns=wine.feature_names)


st.title("This is a web app to predict the class of a wine")
models = {
    "Logistic Regression":logistic_model, "Decision Tree": dtc_model,
     }

#user select the model
selected_model= st.selectbox("Select a model",list(models.keys()))

final_model= models[selected_model]


input_data ={}

for col in wine_df.columns:
    input_data[col] = st.slider(col, min_value= wine_df[col].min(),max_value=wine_df[col].max())

#convert dict to df
input_df = pd.DataFrame([input_data])

st.write(input_df)

input_df_scaled = scaler_model.transform(input_df)

if st.button("Predict"):
    predicted = final_model.predict(input_df_scaled)[0]
    predicted_prob = final_model.predict_proba(input_df_scaled)[0]

#display prediction
    if predicted == 0:
        st.write("The wine is class 1 and from the first region")
    elif predicted == 1:
        st.write("The wine is class 2 and from the second region")
    else:
        st.write("The wine is class 3 and from the third region")


