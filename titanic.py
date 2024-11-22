import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

def main():
    
    loaded = joblib.load('prev_titanic.pkl')
    
    eta = st.slider("quanti anni hai?", 0, 100, 18)
    genere = st.sidebar.selectbox('indica il tuo sesso', 'M', 'F', key='Sex')
    porto = st.sidebar.selectbox('in che zona saresti vissuto dell UK?', 's', 'c', 'q')
    ral = st.slider("calcoliamo la tua classe e il costo del biglietti che saresti stato in grado di pagare in base alla RAL in K Euro", 0, 100, 22)
    
    pclass = st.slider('classe', 1, 3, 2)
    
    fare = st.slider("prezzo biglietto", 0,1000, 30)    
    nucleo = st.slider("Sipsb avresti portato con te il tuo nucleo? se si indica da quante persone è composto altrimenti lascia 0", 0, 10, 0)
    
    parch = st.slider("parch", 0, 5, 0)        
    
    
    utente = {"Age" : [eta],
              "Sex" : [genere],
              "Embarked" : [porto],
              "Pclass" : [pclass],
              "Fare" : [fare],
              "SibSp" : [nucleo],
              "Parch" : [parch]
              }
    
    df_utente = pd.DataFrame(utente)
    
    def sopravvisuto(df_utente):
        pred = loaded.predict(df_utente)[0]
        if pred == 0:
            st.text('caput')
            st.image(Image.open("caput.jpg"), use_container_width=True)
            
        else:
            st.text('che botta di culo')
            st.image(Image.open("rose_old.jpg"), use_container_width=True)
            
            return pred
        
if __name__ == "__main__":
    main()