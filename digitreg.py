# Streamlit app to classify digits
"""
Created on Sun Mar 23 19:24:41 2025

@author: andre
"""


import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import onnxruntime
from sqlalchemy import text

st.set_page_config(page_title="Read Digits")

ort_session = onnxruntime.InferenceSession("inputs/mnist_working.onnx") 
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

conn = st.connection("postgresql", type="sql") 

# define a function for extracting centroid
def prepimg(ima0):
    ima = cv2.cvtColor(ima0, cv2.COLOR_BGR2GRAY)
    nr,nc = ima.shape
    # find the centroid
    cr = round(np.sum(np.dot(ima.transpose(), np.array(range(nr))))/np.sum(ima))
    cc = round(np.sum(np.dot(ima, np.array(range(nc))))/np.sum(ima))
    # find the current box
    rar = np.array(range(nr))[list(np.max(ima, axis=1)>0)]
    rac = np.array(range(nc))[list(np.max(ima, axis=0)>0)]
    # find the new box
    rad = max(cr - np.min(rar), np.max(rar) - cr, cc - np.min(rac), np.max(rac) - cc)
    minr = (cr - rad, 0) if cr > rad else (0, rad - cr)
    maxr = (cr + rad, 0) if cr + rad < nr else (0, cr + rad - nr)
    minc = (cc - rad, 0) if cc > rad else (0, rad - c)
    maxc = (cc + rad, 0) if cc + rad < nc else (0, cc + rad - nc)
    # crop and pad as necessary
    ima = cv2.copyMakeBorder(ima[minr[0]:maxr[0]+1, minc[0]:maxc[0]+1], 
                       minr[1], maxr[1], minc[1], maxr[1],
                       cv2.BORDER_CONSTANT, None, value=0)
    # resize the image
    ima = cv2.resize(ima, (20,20))
    # pad
    ima = cv2.copyMakeBorder(ima, 4, 4, 4, 4, cv2.BORDER_CONSTANT, None, value=0)
    return ima

# define a function for logging information to the database
def logrecog(prval, label):
    with conn.session as session: 
        labeltxt = str(label) if label is not None else "NULL"
        prvaltxt = str(prval) if prval is not None else "NULL"
        sqltxt = "insert into digitreads values (gen_random_uuid(), current_timestamp(0), " + prvaltxt + "," + labeltxt + ");"
        session.execute(text(sqltxt))
        session.commit() 

# define a function for scoring and logging
def recognize_n_record(result, label):
    if result.image_data is not None:
        try:
            ima = prepimg(result.image_data)
            a = ima[np.newaxis,:,:].astype(np.float32) / 255.0
            ort_inputs = {ort_session.get_inputs()[0].name: a}
            ort_outs = ort_session.run(None, ort_inputs)[0][0,:]
            prval = np.argmax(ort_outs)
            conf = 1/np.sum(np.exp(ort_outs - ort_outs[prval]))
            st.session_state['prval'] = str(prval)
            st.session_state['conf'] = f"{100*conf: .0f}%"  
            logrecog(prval, label)
        except:
            st.session_state['prval'] = "Undefined"
            st.session_state['conf'] = "Undefined"             
    
# default session state values
if 'prval' not in st.session_state:
    st.session_state['prval'] = ''
    
if 'conf' not in st.session_state:
    st.session_state['conf'] = ''

# layout
st.header("Digit Recognizer")

left_column, right_column = st.columns(2)

with left_column: 
    canvas_result = st_canvas(
        stroke_color="white",
        stroke_width=10,
        background_color="black",
        height=300,
        width=300,
        drawing_mode="freedraw",
        )

with right_column: 
    prbox = st.write(f"Prediction: {st.session_state['prval']}")
    confbox = st.write(f"Confidence: {st.session_state['conf']}")
    label = st.selectbox( "True Label:", list(range(10)), index=None)
    st.button("Submit", on_click=recognize_n_record, args=[canvas_result, label])

# on_change -- to log
st.header("History") 

# define a function for printing the log
hist = conn.query('select timestamp as "timestamp", predlab as "prediction", truelab as "true label" from digitreads order by timestamp desc limit 10;', ttl="0")
st.dataframe(hist, hide_index=True)
