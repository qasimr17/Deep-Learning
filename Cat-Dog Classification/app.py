import streamlit as st 
import torch 
import torch.nn as nn
from torchvision import models
import inference 

@st.cache
def get_model():
    device = "cuda" if torch.cuda.is_available else "cpu"
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.to(device=device)

    FILE = "./ft_resnet18.pth"
    model.load_state_dict(torch.load(FILE))
    model.eval()
    return model

def make_prediction(model, image):
    pred = inference.predict(model, image)
    return pred 

def show_image(image):
    pass

def main():
    model_resnet18 = get_model()
    st.title("Cat-Dog Image Classification")
    img = st.file_uploader("Please choose an image of a dog or a cat for classification.")

    if img is not None:
        st.image(img)

        if st.button('Make A Prediction!'):
            pred, prob = make_prediction(model_resnet18, img)
            st.text(f"The model predicted that this image is of a {pred}, with probability {(prob * 100):.2f}%")


if __name__ == "__main__":
    main()