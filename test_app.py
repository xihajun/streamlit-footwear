import numpy as np
import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pathlib
import re
from io import StringIO

file_pre = st.sidebar.file_uploader('Upload pre-GOC file:', ('txt'))
file_post = st.sidebar.file_uploader('Upload post-GOC file:', ('txt'))

def read_string(string):
    """load the txt data and return a list of sentences"""
    return re.split('[.;:!,"]',line)[0:-1]


if file_pre and file_post:
    pre_data = StringIO(file_pre.getvalue().decode("utf-8"))
    post_data = StringIO(file_post.getvalue().decode("utf-8"))
    pre_txt, post_txt = pre_data.read(), post_data.read()
    pre_list, post_list = read_string(pre_txt), read_string(post_txt)
    

    score = []
    dic = {}
    for i in pre_list:
        rowscore = []
        for j in post_list:
            dic[(i,j)] = similar(i,j)
            dic[(j,i)] = dic[(i,j)]
            rowscore.append(dic[(i,j)])
        score.append(rowscore)
        
    import matplotlib.pyplot as plt
    fig = plt.imshow(score)
    st.pyplot(fig)