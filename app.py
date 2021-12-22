import streamlit as st
from PIL import Image

st.title("Iris Species Classifier")

sl = st.slider('S e p a l - L e n g t h', 4.3, 7.9, 0.5)
sw = st.slider('S e p a l - W i d t h', 2.0, 4.4, 0.5)
pl = st.slider('P e t a l - L e n g t h', 1.0, 6.9, 0.5)
pw = st.slider('P e t a l - W i d t h', 0.1, 2.5, 0.5)

from sklearn.datasets import load_iris
iris = load_iris()

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(iris.data,iris.target)

op = model.predict([[sl,sw,pl,pw]])

image1 = Image.open('Images/setosa.jpeg')
image2 = Image.open('Images/versicolor.jpg')
image3 = Image.open('Images/virginica.jpg')

if op[0] == 0: st.image(image1)
elif op[0] == 1: st.image(image2)
else: st.image(image3)
