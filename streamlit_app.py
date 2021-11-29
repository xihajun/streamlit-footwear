from io import BytesIO
import PIL.Image
import numpy as np
import os
import requests
import streamlit as st
import subprocess
import tensorflow as tf
import urllib
import zipfile
import pandas as pd
import xml.etree.ElementTree as ElementTree
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, UpSampling2D, Cropping2D, Flatten, Dense, Dropout, BatchNormalization
from skimage import transform, data, io, filters, measure 
import pickle
import matplotlib.pyplot as plt
import keract

from streamlit_juxtapose import juxtapose
import pathlib


'# Where the footwear mask is?'
THIS_FILE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(THIS_FILE_DIR, 'model_2021')
MODEL_FILENAME = os.path.join(MODEL_DIR, 'saved_model.pb')

@st.cache
def download_model_from_web():
    if os.path.isfile(MODEL_FILENAME):
        return
    try:
        os.mkdir(MODEL_DIR)
    except FileExistsError:
        pass

    MODEL_ZIP_URL = (
        'http://www.edu-ing.cn/sslab/'
        'model.zip')
    ZIP_FILE_NAME = 'model.zip'
    ZIP_FILE_PATH = os.path.join(MODEL_DIR, ZIP_FILE_NAME)
    resp = requests.get(MODEL_ZIP_URL, stream=True)

    with open(ZIP_FILE_PATH, 'wb') as file_desc:
        for chunk in resp.iter_content(chunk_size=5000000):
            file_desc.write(chunk)

    zip_file = zipfile.ZipFile(ZIP_FILE_PATH)
    zip_file.extractall(path=MODEL_DIR)

    os.remove(ZIP_FILE_PATH)

    
download_model_from_web()

# load model
# with open("models/ruler_model.pl", "rb") as f:
#     ruler_model = pickle.load(f)

ruler_model = tf.keras.models.load_model(
    "model_2021"
)
# define functions
@st.cache()
def read_file_from_url(url):
    return urllib.request.urlopen(url).read()

def write_image(dg, arr):
    arr = np.uint8(np.clip(arr/255.0, 0, 1)*255)
    dg.image(arr, use_column_width=True)
    return dg

def fetch_img_from_url(url: str) -> PIL.Image:
    img = PIL.Image.open(requests.get(url, stream=True).raw)
    return img

def largeConnectComponent(bw_img):
    labeled_img, num = measure.label(bw_img, neighbors=8, background=0, return_num=True)
    max_label = []
    max_num = 0
    # import pdb; pdb.set_trace()
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > 4000:
            max_num = np.sum(labeled_img == i)
            max_label.append(i)
    mcr = [(labeled_img == large_label) for large_label in max_label]
    if len(mcr) == 1:
      return np.array(mcr).reshape(376,504)
    else:
      return sum(mcr)

def largeConnectComponent_weak(bw_img):
    labeled_img, num = measure.label(bw_img, neighbors=8, background=0, return_num=True)
    max_label = []
    max_num = 0
    # import pdb; pdb.set_trace()
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > 100:
            max_num = np.sum(labeled_img == i)
            max_label.append(i)
    mcr = [(labeled_img == large_label) for large_label in max_label]
    if len(mcr) == 1:
      return np.array(mcr).reshape(376,504)
    else:
      return sum(mcr)

def find_crop_loc(matrix):
    # matrix is the prediction as above
    # task: find the left point, right point, up point and down point
    d,r = matrix.shape
    u = 0
    for i in matrix:
        u+=1
        if sum(i)!=0:
            break
    for i in reversed(matrix):
        d-=1
        if sum(i)!=0:
            break 
    return u,d

def return_big_box(filtered_array):
  up,down = find_crop_loc(filtered_array)
  left,right = find_crop_loc(filtered_array.T)
  return (8*left,8*up,8*right,8*down)

# Sidebar controls:

# Temporary config option to remove deprecation warning.
st.set_option('deprecation.showfileUploaderEncoding', False)

MAX_IMG_WIDTH = 504
MAX_IMG_HEIGHT = 376
DEFAULT_IMAGE_URL = 'https://user-images.githubusercontent.com/25631641/143768680-7b27fbc7-bf20-4f49-8284-437cc850b76a.jpeg'

file_obj = st.sidebar.file_uploader('Choose an image:', ('jpg', 'jpeg'))

import uuid

IMG1 = str(uuid.uuid4())+".png"
IMG2 = str(uuid.uuid4())+".png"
IMG3 = str(uuid.uuid4())+".png"
IMG4 = str(uuid.uuid4())+".png"

if not file_obj:
    file_obj = BytesIO(read_file_from_url(DEFAULT_IMAGE_URL))
# else:
#     IMG1 = "test1.png"
#     IMG2 = "test2.png"

img_in = np.asarray(PIL.Image.open(file_obj))

write_image(st.sidebar, img_in)

'## Output'

# input ori to small size
def ori_small_img(img):
  # load the training img
  (width,height,depth) = img.shape
  img = np.array(img.reshape((width,height,depth)))# / 255
  return img, transform.resize(img, (376,504))

ori, image = ori_small_img(img_in)

image = np.expand_dims(image, axis=0)
# Train the model and do the prediction
prediction = ruler_model.predict(np.array(image))

(width,height,depth) = (376,504,3)

#print(prediction.squeeze().shape)
predictions_footwear_img = prediction.squeeze()[:,:,0].reshape(width,height)
predictions_bw_img = prediction[0][:,:,1].reshape(width,height)

st.image(predictions_bw_img)

selected_areas = largeConnectComponent(np.rint(predictions_footwear_img))

labeled_img, num = measure.label(np.rint(predictions_footwear_img), neighbors=8, background=0, return_num=True)

might_wrong = []
maker_length_all = []

find_loc = sum(predictions_bw_img.T)
locations = np.where(find_loc >= max(find_loc))

# col_start to col_end should be the marker 
col_start = min(locations[0]) - 6
col_end = max(locations[0]) + 6
selected_marker_row = sum(predictions_bw_img[col_start:col_end])
mean = np.mean(selected_marker_row)
two_marker_loc = np.where(selected_marker_row>=mean)[0]
# 两个连续的数列
marker = range(min(two_marker_loc),max(two_marker_loc)+1)
med = list(set(marker)-set(two_marker_loc))

if len(med) == len(list(range(min(med),max(med)+1))):
    print("yes this should be the marker")
    maker_center = np.median(med)
    maker_length = max(two_marker_loc) - min(two_marker_loc)
    maker_width = maker_length/14
    might_wrong.append(0)
else:
    print("something wrong, trying to fix it")
    # plt.figure()
    # plt.plot(selected_marker_row)
    selected_marker_row[selected_marker_row>np.mean(selected_marker_row)]=1
    selected_marker_row[selected_marker_row!=1] = 0
    sum_bag = test(list(selected_marker_row))
    center_pixels = np.where(np.abs(sum_bag) == max(np.abs(sum_bag)))[0]
    maker_center = np.median(center_pixels)
    maker_length = len(center_pixels)*4
    maker_width = maker_length/14
    might_wrong.append(1)

maker_length_all.append(maker_length)

if np.median(locations[0])>376/2:
    selected_areas[int(np.median(locations[0])-maker_width):,:] = 0
    # plt.figure()
    # plt.imshow(selected_areas)
    # 在下就在左边
    try:
        selected_areas[:,:int(maker_center-maker_length/140*150+maker_width)] = 0
        # plt.figure()
        # plt.imshow(selected_areas)
    except:
        print(maker_center-maker_length/140*150+maker_width,np.median(locations[0])-maker_width-maker_length/140*85)
else:
    selected_areas[:int(np.median(locations[0])+maker_width),:] = 0
    # plt.figure()
    # plt.imshow(selected_areas)
    # 在上就在右边
    try:
        selected_areas[:,int(maker_center+maker_length/140*150-maker_width):] = 0
        # plt.figure()
        # plt.imshow(selected_areas)
    except:
        print(maker_center+maker_length/140*150-maker_width,np.median(locations[0])+maker_width+maker_length/140*85)
might_wrong = []
maker_length_all = []

find_loc = sum(predictions_bw_img.T)
locations = np.where(find_loc >= max(find_loc))

# col_start to col_end should be the marker 
col_start = min(locations[0]) - 6
col_end = max(locations[0]) + 6
selected_marker_row = sum(predictions_bw_img[col_start:col_end])
mean = np.mean(selected_marker_row)
two_marker_loc = np.where(selected_marker_row>=mean)[0]
# 两个连续的数列
marker = range(min(two_marker_loc),max(two_marker_loc)+1)
med = list(set(marker)-set(two_marker_loc))

if len(med) == len(list(range(min(med),max(med)+1))):
    print("yes this should be the marker")
    maker_center = np.median(med)
    maker_length = max(two_marker_loc) - min(two_marker_loc)
    maker_width = maker_length/14
    might_wrong.append(0)
else:
    print("something wrong, trying to fix it")
    # plt.figure()
    # plt.plot(selected_marker_row)
    selected_marker_row[selected_marker_row>np.mean(selected_marker_row)]=1
    selected_marker_row[selected_marker_row!=1] = 0
    sum_bag = test(list(selected_marker_row))
    center_pixels = np.where(np.abs(sum_bag) == max(np.abs(sum_bag)))[0]
    maker_center = np.median(center_pixels)
    maker_length = len(center_pixels)*4
    maker_width = maker_length/14
    might_wrong.append(1)

maker_length_all.append(maker_length)

if np.median(locations[0])>376/2:
    selected_areas[int(np.median(locations[0])-maker_width):,:] = 0
    # plt.figure()
    # plt.imshow(selected_areas)
    # 在下就在左边
    try:
        selected_areas[:,:int(maker_center-maker_length/140*150+maker_width)] = 0
        # plt.figure()
        # plt.imshow(selected_areas)
    except:
        print(maker_center-maker_length/140*150+maker_width,np.median(locations[0])-maker_width-maker_length/140*85)
else:
    selected_areas[:int(np.median(locations[0])+maker_width),:] = 0
    # plt.figure()
    # plt.imshow(selected_areas)
    # 在上就在右边
    try:
        selected_areas[:,int(maker_center+maker_length/140*150-maker_width):] = 0
        # plt.figure()
        # plt.imshow(selected_areas)
    except:
        print(maker_center+maker_length/140*150-maker_width,np.median(locations[0])+maker_width+maker_length/140*85)

STREAMLIT_STATIC_PATH = (
    pathlib.Path(st.__path__[0]) / "static"
)  # at venv/lib/python3.9/site-packages/streamlit/static


try:
    print("removing")
    shutil.rmtree(STREAMLIT_STATIC_PATH / IMG1)
    shutil.rmtree(STREAMLIT_STATIC_PATH / IMG2)
except:
    print("nothing")

# from here
selectedfig = plt.figure()
plt.imshow(selected_areas,interpolation="nearest")
plt.axis('off')
plt.imsave(STREAMLIT_STATIC_PATH / IMG1,labeled_img)
plt.close()


testfig = plt.figure()
plt.imshow(selected_areas,interpolation="nearest")
plt.axis('off')
plt.imsave(STREAMLIT_STATIC_PATH / IMG2, selected_areas)
plt.close()

'## Output'

box = return_big_box(selected_areas)
img = PIL.Image.fromarray(ori)
cropped = np.array(img.crop(box))
st.image(cropped)

'## Largest Connected Component'
'The model returns things with noise. Because shoes have two main parts, why not select largest components for the output? It works for most of our test sets. However, this one is not working, lets go down and see how did I fix that issue'
juxtapose(IMG1, IMG2)



# from here
selectedfig = plt.figure()
plt.imshow(selected_areas,interpolation="nearest")
plt.axis('off')
plt.imsave(STREAMLIT_STATIC_PATH / IMG3,predictions_footwear_img)
plt.close()


testfig = plt.figure()
plt.imshow(selected_areas,interpolation="nearest")
plt.axis('off')
plt.imsave(STREAMLIT_STATIC_PATH / IMG4, predictions_bw_img)
plt.close()


'## Ruler Detector'
'So by redesigning the model adding ruler label and footwear mask label, this model could detect footwear and even estimate the length of the shoe.'

find_loc = sum(predictions_bw_img.T)
locations = np.where(find_loc >= max(find_loc))
# col_start to col_end should be the marker 
col_start = min(locations[0]) - 5
col_end = max(locations[0]) + 5

selected_marker_row = sum(predictions_bw_img[col_start:col_end])
mean = np.mean(selected_marker_row)
two_marker_loc = np.where(selected_marker_row>=mean)[0]
# 两个连续的数列
marker = range(min(two_marker_loc),max(two_marker_loc)+1)
med = list(set(marker)-set(two_marker_loc))
if len(med) == len(list(range(min(med),max(med)+1))):
    print("yes this should be the marker")
    maker_center = np.median(med)
    maker_length = max(two_marker_loc) - min(two_marker_loc)
    maker_width = maker_length/14
# plt.plot(sum(predictions_bw_img[col_start:col_end]))
    print(np.median(locations),maker_center)
    fig = plt.figure()
    plt.imshow(predictions_bw_img)
    plt.plot(maker_center,np.median(locations[0]),'bo')
    plt.plot(maker_center,np.median(locations[0])+maker_width,'ro')
    plt.plot(maker_center,np.median(locations[0])-maker_width,'go')
else:
    print("something wrong, trying to fix it")
    selected_marker_row[selected_marker_row>np.mean(selected_marker_row)]=1
    selected_marker_row[selected_marker_row!=1] = 0
    sum_bag = test(list(selected_marker_row))
    center_pixels = np.where(np.abs(sum_bag) == max(np.abs(sum_bag)))[0]
    print(center_pixels)
    maker_center = np.median(center_pixels)
    maker_length = len(center_pixels)*4
    maker_width = maker_length/14

    print(np.median(locations),maker_center)
    fig = plt.figure()
    plt.imshow(predictions_bw_img)
    plt.plot(maker_center,np.median(locations[0]),'bo')
    plt.plot(maker_center,np.median(locations[0])+maker_width,'ro')
    plt.plot(maker_center,np.median(locations[0])-maker_width,'go')
if np.median(locations[0])>376/2:
    # 在下就在左边
    try:
        plt.plot(maker_center-maker_length/140*150,np.median(locations[0])-maker_width-maker_length/140*85,'ro')
        plt.plot(maker_center-maker_length/140*150+maker_width,np.median(locations[0])-maker_width-maker_length/140*85,'bo')
    except:
        print(maker_center-maker_length/140*150+maker_width,np.median(locations[0])-maker_width-maker_length/140*85)
else:
    # 在上就在右边
    try:
        plt.plot(maker_center+maker_length/140*150,np.median(locations[0])+maker_width+maker_length/140*85,'ro')
        plt.plot(maker_center+maker_length/140*150-maker_width,np.median(locations[0])+maker_width+maker_length/140*85,'bo')
    except:
        print(maker_center+maker_length/140*150-maker_width,np.median(locations[0])+maker_width+maker_length/140*85)

st.pyplot(fig)

juxtapose(IMG3, IMG4)



