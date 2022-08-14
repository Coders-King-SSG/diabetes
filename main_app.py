import streamlit as st
import numpy as np
import pandas as pd
import home
import predict
import visualise

# Configure your home page by setting its title and icon that will be displayed in a browser tab.
st.set_page_config(page_title = 'Early Diabetes Prediction Web App',
                    page_icon = 'icon.png',
                    layout = 'wide',
                    initial_sidebar_state = 'auto'
                    )

# Loading the dataset.
@st.cache()
def load_data():
    # Load the Diabetes dataset into DataFrame.

    df = pd.read_csv('data.csv')
    df.head()

    # Rename the column names in the DataFrame.
    df.rename(columns = {"BloodPressure": "Blood_Pressure",}, inplace = True)
    df.rename(columns = {"SkinThickness": "Skin_Thickness",}, inplace = True)
    df.rename(columns = {"DiabetesPedigreeFunction": "Pedigree_Function",}, inplace = True)

    df.head() 

    return df

diabetes_df = load_data()
pages_dict = {"Home": home,"Visualise Data": visualise,"Predict": predict}
st.sidebar.title('Navigation')
ch = st.sidebar.radio('Go to', pages_dict.keys())
if ch =='Home':
	home.app()
else:
	pages_dict[ch].app(diabetes_df)