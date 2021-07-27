from PIL import Image
import numpy as np
import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn import svm
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from skimage.feature import greycomatrix, greycoprops
import pickle
import xgboost as xgb
from xgboost import XGBClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ltoi = {"stage_0": 0, "stage_5": 1, "stage_6": 2, "stage_8": 3}
itol = {0: "stage_0", 1: "stage_5", 2: "stage_6", 3: "stage_8"}
bins = 14
w = 200
h = 200
p_no = 5
p_size = 75
limit = 30
stages = list(ltoi.keys())
f_size = 50

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


# cnn = model_loader('cnn_model.pickle')
params = {
    "gamma": 3.84122636666823,
    "max_depth": 36.46280220719902,
    "min_child_weight": 3.477294203777718,
    "n_estimators": 86.7722952613406,
    "num_boost_round": 962.1524651948235,
    "reg_alpha": 0.17400408126064204,
    "reg_lambda": 0.6379029525314178,
}

# cnnxgb = model_loader("cnnxgb_model.pickle")
# xgb_model = xgb.Booster()
# xgb_model = XGBClassifier(**params)
# xgb_model.load_model("cnnxgb_model.json")
booster = xgb.Booster()
booster.load_model("cnnxgb_model.bin")
Fit = model_loader("scaler_fit.pickle")
mlfit = model_loader("ml_fit.pickle")
gnb_model = model_loader("gnb_model.pickle")
knn_model = model_loader("knn_model.pickle")
svm_model = model_loader("svm_model.pickle")
cnn_model = model_loader("cnn_model.pickle")
mean_tens = torch.tensor([0.485, 0.456, 0.406])
std_tens = torch.tensor([0.229, 0.224, 0.225])


def segment_image(img, cspace=None):
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if cspace == "HSV":
        cimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if cspace == "LAB":
        cimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if cspace == None:
        # cimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cimg = img
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gimg = cv2.medianBlur(gimg, 3)
    adap_thresh = cv2.adaptiveThreshold(
        gimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 1
    )
    contours, hierarchy = cv2.findContours(
        adap_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_areas = sorted(contours, key=cv2.contourArea)
    mask = np.zeros(gimg.shape, np.uint8)
    gimg_contour = cv2.drawContours(
        mask, [largest_areas[-1]], 0, (255, 255, 255, 255), -1
    )
    gimask = gimg_contour > 0
    canvas = np.full_like(cimg, 0, dtype=np.uint8)
    canvas[gimask] = cimg[gimask]
    return canvas, gimg, gimg_contour


class seg(object):
    def __call__(self, img):
        img = np.array(img)
        canvas, _, _ = segment_image(img)
        return Image.fromarray(canvas)

    def __repr__(self):
        return self.__class__.__name__ + "()"


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def getpatches(gimg, p_no, p_size=None, limit=None):
    # print(gimg.shape)
    # cv2_imshow(gimg)
    if p_size == None:
        p_size = int(0.2 * w)
    if limit == None:
        limit = int(0.1 * p_size * p_size)
    patches = []
    counts = []
    while len(patches) < p_no:
        # x = np.random.randint(int(w/2),w-p_size+1)
        # y = np.random.randint(w-p_size+1)
        x = np.random.randint(w - p_size + 1)
        y = np.random.randint(w - p_size + 1)
        patch = gimg[x : x + p_size, y : y + p_size]
        count = np.count_nonzero(patch > 220)
        # cv2_imshow(patch)
        if count > limit:
            x = np.random.randint(int(w / 2), w - p_size + 1, size=1)
            y = np.random.randint(w - p_size + 1, size=1)
        else:
            # cv2_imshow(patch)
            patches.append(patch)
            counts.append(count)
    return patches


def extract_features(img):

    canvas, gimg, gimg_contour = segment_image(img, "HSV")
    hist = cv2.calcHist(
        [canvas], [0, 1, 2], gimg_contour, [bins, bins, bins], [0, 210, 0, 210, 0, 210]
    )
    hist = cv2.normalize(hist, hist).flatten()
    patches = getpatches(gimg, p_no)
    texture_patch = np.zeros(5)
    for patch, j in zip(patches, range(p_no)):
        glcm = greycomatrix(
            patch, distances=[5], angles=[0], levels=256, symmetric=True, normed=True
        )
        texture_patch[0] += greycoprops(glcm, "dissimilarity")[0, 0]
        texture_patch[1] += greycoprops(glcm, "correlation")[0, 0]
        texture_patch[2] += greycoprops(glcm, "homogeneity")[0, 0]
        texture_patch[3] += greycoprops(glcm, "energy")[0, 0]
        texture_patch[4] += greycoprops(glcm, "contrast")[0, 0]
    temp = np.hstack((hist, texture_patch / p_no))
    return temp


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
    preds = ""
    pred = []
    with torch.no_grad():
        _ = cnn_model(x.to(device))
    activations = activation["fc1"]
    test_xgb_f = torch.cat((test_xgb_f, activations.detach().cpu()), dim=0)
    test_xgb_f = test_xgb_f.detach().numpy()
    ml_f = np.empty([0, bins * bins * bins + 5])
    # print(pth)
    # img = cv2.imread(pth)
    img = img.convert("RGB")
    open_cv_image = np.array(img)
    # Convert RGB to BGR
    img = open_cv_image[:, :, ::-1].copy()
    temp = extract_features(img)
    ml_f = np.vstack((ml_f, temp))
    ml_f_std = mlfit.transform(ml_f)
    with torch.no_grad():
        y = cnn_model(x).max(1)[1].item()
    st.write("CNN prediction ", itol[y])
    y = booster.predict(xgb.DMatrix(test_xgb_f))
    # print(y)
    st.write("CNN XGB prediction ", itol[y[0].argmax()])
    y = svm_model.predict(ml_f_std)
    st.write("SVM prediction ", itol[y[0]])
    y = gnb_model.predict(ml_f)
    st.write("GNB prediction ", itol[y[0]])
    y = knn_model.predict(ml_f_std)
    st.write("KNN prediction ", itol[y[0]])


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    Generate_pred = st.button("Predict Ripeness Stage..")
    if Generate_pred:
        predict_stage(file)
        # st.text("Probability (0: Unripe, 1: Overripe, 2: Ripe")
        # st.write(prediction)
