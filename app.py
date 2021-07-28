from PIL import Image
import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
from sklearn.preprocessing import StandardScaler
import pickle
import xgboost as xgb
from xgboost import XGBClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ltoi = {"Unripe :(": 0, "Ripe :)": 1, "Overripe -.-": 2}
itol = {0: "Unripe :(", 1: "Ripe :)", 2: "Overripe -.-"}
w = 200
h = 200
stages = list(ltoi.keys())
f_size = 10

st.write(
    """
# Banana Ripeness Detection
"""
)
st.write("A Image Classification Web App That Detects the Ripeness Stage of Banana")

file = st.file_uploader("Please Upload an image of banana", type=["jpg", "png", "jpeg"])


def model_loader(pth):
    f = open(pth, "rb")
    model = pickle.load(f)
    f.close()
    return model


cnn_model = model_loader("cnn_model.pickle")


booster = xgb.Booster()
booster.load_model("cnnxgb_model.bin")
Fit = model_loader("scaler_fit.pickle")

mean_tens = torch.tensor([0.485, 0.456, 0.406])
std_tens = torch.tensor([0.229, 0.224, 0.225])


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def predict_stage(pth):
    img = Image.open(pth).convert("RGB")
    mods = transforms.Compose(
        [
            transforms.Resize((w, h)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_tens.tolist(), std=std_tens.tolist()),
        ]
    )
    x = mods(img).unsqueeze(dim=0)
    test_xgb_f = torch.empty(0, f_size)
    list(cnn_model.classifier.children())[3].register_forward_hook(
        get_activation("fc1")
    )
    cnn_model.eval()
    with torch.no_grad():
        y = cnn_model(x.to(device))
    activations = activation["fc1"]
    test_xgb_f = torch.cat((test_xgb_f, activations.detach().cpu()), dim=0)
    test_xgb_f = test_xgb_f.detach().numpy()
    test_xgb_f = Fit.transform(test_xgb_f)
    y = y.max(1)[1].item()
    st.write("CNN prediction: ", itol[y])
    del y
    torch.cuda.empty_cache()
    y = booster.predict(xgb.DMatrix(test_xgb_f))
    st.write("CNN XGB prediction: ", itol[y[0].argmax()])
    del y


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    Generate_pred = st.button("Predict Ripeness Stage..")
    if Generate_pred:
        predict_stage(file)
