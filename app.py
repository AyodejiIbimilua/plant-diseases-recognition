import streamlit as st
## importing of need libraries to working environment
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import ImageFile
import PIL
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler


PAGE_CONFIG = {"page_title":"Plant Disease Recognition","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

class_names = ['Apple Scab',
 'Apple Black Rot',
 'Apple Cedar Rust',
 'Healthy Apple',
 'Healthy Bluberry',
 'Cherry Powdery Mildew',
 'Healthy Cherry',
 'Maize Gray Leaf Spot',
 'Maize Common Rust',
 'Maize Northern Leaf Blight',
 'Healthy Maize',
 'Grape Black Rot',
 'Grape Black Measles',
 'Grape Leaf Blight',
 'Grape Healthy',
 'Orange Citrus Greening',
 'Peach Bacterial Spot',
 'Healthy Peach',
 'Pepper Bell Bacterial Spot',
 'Healthy Pepper Bell',
 'Potato Early Blight',
 'Potato Late Blight',
 'Healthy Potato',
 'Healthy Raspberry',
 'Healthy Soybean',
 'Squash Powdery Mildew',
 'Strawberry Leaf Scorch',
 'Healthy Strawberry',
 'Tomato Bacterial Spot',
 'Tomato Early Blight',
 'Tomato Late Blight',
 'Tomato Leaf Mold',
 'Tomato Septoria Leaf Spot',
 'Tomato Spider Mites',
 'Tomato Target Spot',
 'Tomato Yellow Leaf Curl Virus',
 'Tomato Mosaic Virus',
 'Healthy Tomato']


class DiseaseClassifier(nn.Module):
    def __init__(self):
        super(DiseaseClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(128*7*7, 20000)
        self.fc2 = nn.Linear(20000, 38)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
       
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, 7*7*128)
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        x = self.fc2(x)
        return x

@st.cache
def disease_classifier():
    disease_classifier = DiseaseClassifier()
    disease_classifier.load_state_dict(torch.load("model_scratch.pt", map_location=torch.device('cpu')))
    return disease_classifier

# loads the image and performs transformation on it
def load_input_image(img_path):
    image = PIL.Image.open(img_path).convert("RGB")
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    image = prediction_transform(image).unsqueeze(0)
    return image

# runs prediction on the image while returning the class name and index number
def predict_image_class(model, class_names, img_path):
    img = load_input_image(img_path)
    model = model.cpu()
    model.eval()
    idx = torch.argmax(model(img))
    return class_names[idx], idx

def run_app(img_path):
    ## runs the actual prediction while plotting and displaying its prediction
    img = PIL.Image.open(img_path).convert('RGB')
    name, ids = predict_image_class(disease_classifier(), class_names, img_path) 
    
    return ids  

def main():
    st.markdown(
    '''
    <h1 style="color:blue, color:blue">
        Plant Disease Recognition with Deep Learning
    </h1>
    ''',
    unsafe_allow_html=True)

    menu = ["Home", "About Project", "Student Info"]

    choice = st.sidebar.selectbox('Menu', menu)
    if choice == 'Home':
        st.subheader("Make Predictions")


        uploaded_file = st.file_uploader("To classify a disease, upload an image of the Leaf of the crop", type="jpg")
        if uploaded_file is not None:
            image = PIL.Image.open(uploaded_file)
            st.image(image, width=400, height=150)

            ids = run_app(uploaded_file)
            for i, j in enumerate(class_names):
                if ids == i:
                    st.subheader("Classification Result: {}".format(j))

    if choice == "About Project":
        st.subheader("About Project")
        st.write("This project demonstrates how a convolutional neural network can help to identify plant diseases using \
            the image of the leaves. This model is able to identify thirty-eight(38) different classes of diseases ranging from thirteen different types of crops.")
        st.write("")
        st.write("Currently able to identify from the following 13 range of crops:")
        st.write("1. Apple")
        st.write("2. Blueberry")
        st.write("3. Cherry")
        st.write("4. Grape")
        st.write("5. Maize")
        st.write("6. Orange")
        st.write("7. Peach")
        st.write("8. Pepper")
        st.write("9. Potato")
        st.write("10. Raspberry")
        st.write("11. Soybean")
        st.write("12. Strawberry")
        st.write("13. Tomato")
    if choice == "Student Info":
        st.subheader("Student Info")
        st.write("Naame: Ibimilua Ayodeji Oluwatosin")
        st.write("Matric NO: 158540054")
        st.write("Department: Electrical and Electronic Engineering")
        st.write("Faculty: Engineering")
        st.write("Supervisor: Dr. O. Adeitan")
        st.write("Year: 2020")
        st.write("Ekiti State University")

        



if __name__ == '__main__':
	main()