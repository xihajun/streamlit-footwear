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


file_pre = st.sidebar.file_uploader('Upload pre-GOC file:', ('txt'), key="1"))
file_post = st.sidebar.file_uploader('Upload post-GOC file:', ('txt'), key="2"))

if file_pre and file_post:
    pre_data = StringIO(file_pre.getvalue().decode("utf-8"))
    post_data = StringIO(file_post.getvalue().decode("utf-8"))
    pre_data = pre_data.read()
    post_data = pre_data.read()
    st.write(post_data)
