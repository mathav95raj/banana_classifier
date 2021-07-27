from PIL import Image
import numpy as np
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


# model.classifier[6].out_features = len(ltoi)
cnn_model = model_loader("cnn_model.pickle")
# params = {
#     "gamma": 3.84122636666823,
#     "max_depth": 36.46280220719902,
#     "min_child_weight": 3.477294203777718,
#     "n_estimators": 86.7722952613406,
#     "num_boost_round": 962.1524651948235,
#     "reg_alpha": 0.17400408126064204,
#     "reg_lambda": 0.6379029525314178,
# }


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
    img = Image.open(pth)
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
    with torch.no_grad():
        _ = cnn_model(x.to(device))
    activations = activation["fc1"]
    test_xgb_f = torch.cat((test_xgb_f, activations.detach().cpu()), dim=0)
    test_xgb_f = test_xgb_f.detach().numpy()
    test_xgb_f = Fit.transform(test_xgb_f)
    with torch.no_grad():
        y = cnn_model(x).max(1)[1].item()
    st.write("CNN prediction: ", itol[y])
    y = booster.predict(xgb.DMatrix(test_xgb_f))
    st.write("CNN XGB prediction: ", itol[y[0].argmax()])


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    Generate_pred = st.button("Predict Ripeness Stage..")
    if Generate_pred:
        predict_stage(file)
