import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Player Market Value Prediction App

This app predicts the **Player Market Value**!
Keep in mind that this model is trained with data from the most valuable players in the top 5 leagues. So the market values will start from 20M approximately.
""")
st.write('---')

# Loads the Boston House Price Dataset
df = pd.read_csv('data.csv')
X = df.drop(columns=['Market value', 'Unnamed: 0'])
Y = df['Market value']
#print(X)
#print(Y)
X.rename(columns={'Gls+Ast':'Gls_Ast', 'Passes Completed':'Passes_Completed','Passes Attempted':'Passes_Attempted'},inplace=True)
print(X)
print(Y)
X = X.astype(int)
print(X[3:].describe())
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    Age = st.sidebar.slider('Age', 16,50,20)
    Min = st.sidebar.slider('Min', 0,10000,1000)
    Gls = st.sidebar.slider('Gls', 0,90,10)
    Ast = st.sidebar.slider('Ast', 0,90,10)
    Gls_Ast = st.sidebar.slider('Gls+Ast', 0,120,15)
    Tackle = st.sidebar.slider('Tackle', 0,200,30)
    Press = st.sidebar.slider('Press', 0,5000,500)
    Blocks = st.sidebar.slider('Blocks', 0,300,50)
    Passes_Completed = st.sidebar.slider('Passes Completed', 0,10000,1500)
    Passes_Attempted = st.sidebar.slider('Passes Attempted', 0,10000,2000)
    data = {'Age': Age,
            'Min': Min,
            'Gls': Gls,
            'Ast': Ast,
            'Gls_Ast': Gls_Ast,
            'Tackle': Tackle,
            'Press': Press,
            'Blocks': Blocks,
            'Passes_Completed': Passes_Completed,
            'Passes_Attempted': Passes_Attempted
            }
    features = pd.DataFrame(data, index=[0])
    return features

df_ml = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df_ml)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df_ml)

st.header('Prediction of Market Value')
st.write(prediction)
st.write('---')

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')

st.write("""
         If you want to contribute to this model.
         Contact me:
         **dllanessuarez@gmail.com or 
         llanessuarez17@gmail.com**
         """)