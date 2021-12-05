from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
# from sklearn.preprocessing import StandardScaler
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ltoi = {"Unripe :(": 0, "Underripe :/": 1, "Ripe :)": 2, "Overripe -.-": 3}
itol = {0: "Unripe :(", 1: "Underripe :/", 2: "Ripe :)", 3: "Overripe -.-"}
w = 200
h = 200
stages = list(ltoi.keys())
f_size = 65

st.write(
    """
# Banana Ripeness Detection
"""
)
st.write("A Image Classification Web App That Detects the Ripeness Stage of Banana")

file = st.file_uploader("Please Upload an image of banana", type=[
                        "jpg", "png", "jpeg"])


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def model_loader(pth):
    f = open(pth, "rb")
    model = CPU_Unpickler(f).load()
    f.close()
    return model


cnn_model = model_loader("cnn_model.pickle")


booster = xgb.Booster()
booster.load_model("cnnxgb_model.bin")
# Fit = model_loader("scaler_fit.pickle")
# lf = model_loader("lda_model.pickle")
# mean_tens = torch.tensor([0.7011, 0.6698, 0.4972])
# std_tens = torch.tensor([0.1924, 0.2086, 0.2690])
mean_tens = torch.tensor([0.0048, 0.0302, 0.0067])
std_tens = torch.tensor([1.4197, 1.3056, 1.3567])

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
            transforms.Normalize(mean=mean_tens.tolist(),
                                 std=std_tens.tolist()),
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
    # test_xgb_f = Fit.transform(test_xgb_f)
    # test_xgb_f = lf.transform(test_xgb_f)
    y = y.max(1)[1].item()
    # softmax = nn.Softmax(dim=-1)
    # y = softmax(y)
    # y = torch.round(y[0] * 10 ** 2) / (10 ** 2)
    st.write("CNN prediction: ", itol[y])
    # st.write("Unripe\t"+str(y[0].item())+"%")
    # st.write("Underripe "+str(y[1].item())+"%")
    # st.write("Ripe\t"+str(y[2].item())+"%")
    # st.write("Overripe "+str(y[3].item())+"%")
    del y
    torch.cuda.empty_cache()
    y = booster.predict(xgb.DMatrix(test_xgb_f))
    # y = booster.predict(xgb.DMatrix(test_xgb_f)).tolist()[0]*100
    # y = ['%.2f' % elem for elem in y]
    st.write("CNN XGB prediction: ", itol[y[0].argmax()])
    # y = torch.round(y[0] * 10^2) / (10^2)
    # st.write("CNN XGB prediction: \n")
    # st.write("Unripe "+str(y[0])+"%")
    # st.write("Underripe "+str(y[1])+"%")
    # st.write("Ripe "+str(y[2])+"%")
    # st.write("Overripe "+str(y[3])+"%")
    del y


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    Generate_pred = st.button("Predict Ripeness Stage..")
    if Generate_pred:
        predict_stage(file)
