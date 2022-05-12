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


'# re'
'A footwear mark hunter - footwear mark segmentation!'



file_pre = st.sidebar.file_uploader('Upload pre-GOC file:', ('txt'))
file_post = st.sidebar.file_uploader('Upload post-GOC file:', ('txt'))

if file_pre and file_post:
    pre_data = StringIO(file_pre.getvalue().decode("utf-8"))
    post_data = StringIO(file_post.getvalue().decode("utf-8"))
    pre_data = pre_data.read()
    post_data = pre_data.read()
    st.write(post_data)
