import Model
import Helper
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2


st.set_page_config(page_title="ML: Smiley App", page_icon = '🙂')


st.write(
"""
# Model
Draw a happy or sad smiley face and click the button to see the model
prediction.
"""
)

# Drawing Canvas
canvas_result = st_canvas(
    stroke_width=10,
    background_color="White",
    width = 300,
    height= 300,
)

# Flow:
# Click Button --> Collect image data -->
# Reduce Image --> Make Prediction -->
# Print results on screen
if st.button("Predict"):
    data = Helper.image_reducer(canvas_result.image_data.astype('float32'))
    predictions = My_Smiley_Model.make_prediction(data)
    sad = predictions["Sad"]
    happy = predictions["Happy"]
    if sad > happy:
        st.write("Sad")
        st.write(sad)
    else:
        st.write("Happy")
        st.write(happy)
