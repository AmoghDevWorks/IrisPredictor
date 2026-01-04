from sklearn.datasets import load_iris 
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

@st.cache_data
def load_dataset():
    data = load_iris()
    df = pd.DataFrame(data.data,columns=data.feature_names)
    df['Species'] = data.target
    return df,data.target_names

df,target_names = load_dataset()

x = df.iloc[:,:-1]
y = df.iloc[:,-1]

model = RandomForestClassifier(random_state=42)
model.fit(x,y)

st.title('Iris Flower Classifier')
st.sidebar.header('Input Features')

sepal_length = st.sidebar.slider(
    "Sepal length (cm)", float(x["sepal length (cm)"].min()), float(x["sepal length (cm)"].max())
)
sepal_width = st.sidebar.slider(
    "Sepal width (cm)", float(x["sepal width (cm)"].min()), float(x["sepal width (cm)"].max())
)
petal_length = st.sidebar.slider(
    "Petal length (cm)", float(x["petal length (cm)"].min()), float(x["petal length (cm)"].max())
)
petal_width = st.sidebar.slider(
    "Petal width (cm)", float(x["petal width (cm)"].min()), float(x["petal width (cm)"].max())
)

input_data = pd.DataFrame(
    [[sepal_length,sepal_width,petal_length,petal_width]],
    columns = x.columns
)

# Prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display results
st.subheader("Prediction")
st.write(f" **Predicted Species:** `{target_names[prediction[0]]}`")

st.subheader("Prediction Probability")
st.write(pd.DataFrame(prediction_proba, columns=target_names))

if st.checkbox("Show Dataset"):
    st.dataframe(df)