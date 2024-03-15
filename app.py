import gradio as gr
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

data = datasets.load_iris()
X = data.data
Y = data.target
model= DecisionTreeClassifier()
model.fit(X, Y)


def iris(sepal_length ,sepal_width, petal_length, petal_width):
    prediction = model.predict([[sepal_length ,sepal_width, 
                                petal_length, petal_width]])
    prediction= data.target_names[prediction]
    return prediction

#create input and output objects
#input object1
input1 = gr.Number(label="sepal length (cm)")
#input object 2
input2 = gr.Number(label="sepal width (cm)")
#input object3
input3 = gr.Number(label="petal length (cm)")
#input object 3
input4 = gr.Number(label="petal width (cm)")
#output object
output = gr.Textbox(label= "Name of Species") 

#create interface
gui = gr.Interface(fn=iris,
                   inputs=[input1, input2, input3, input4],
                   outputs=output).launch()
