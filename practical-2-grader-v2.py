import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

st.title("Practical 2 - Working With Coordinates")
st.info("""
INSTRUCTIONS
- Select the grades csv file.
- Select the student submission
- Grade by using the appropriate checkboxes
- Download the graded copy and save in a folder (you will be sending these to the senior tutor)
- Add the grade in the appropriate column on a separate Excel worksheet.
""")

gc = st.file_uploader("Select the grades CSV file", type = "csv")
global df
df = pd.read_csv(gc)

image = st.file_uploader("Select the student submission", type = ["png", "jpg"])

if image and gc is not None:
    img_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv.imdecode(img_bytes, 1)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    parameters =  cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    ids = np.concatenate(ids, axis=0).tolist()
    WIDTH = 712
    HEIGHT = 972
    aruco_top_left = corners[ids.index(0)]
    aruco_top_right = corners[ids.index(1)]
    aruco_bottom_right = corners[ids.index(2)]
    aruco_bottom_left = corners[ids.index(3)]
    point1 = aruco_top_left[0][0]
    point2 = aruco_top_right[0][1]
    point3 = aruco_bottom_right[0][2]
    point4 = aruco_bottom_left[0][3]
    working_image = np.float32([[point1[0], point1[1]],
                                [point2[0], point2[1]],
                                [point3[0], point3[1]],
                                [point4[0], point4[1]]])
    working_target = np.float32([[0, 0],
                                 [WIDTH, 0],
                                 [WIDTH, HEIGHT],
                                 [0, HEIGHT]])
    transformation_matrix = cv.getPerspectiveTransform(working_image, working_target)
    warped_img = cv.warpPerspective(img_gray, transformation_matrix, (WIDTH, HEIGHT))
    details = warped_img[0:280, 0:972]
    q1 = warped_img[352:933,0:712]
    q1= cv.rectangle(q1, (184, 79), (218, 113), (0, 0, 255), 1, 8, 0)
    q1 = cv.rectangle(q1, (275,377), (309,411), (0, 0, 255), 1, 8, 0)
    q1 = cv.rectangle(q1, (556,459), (583,492), (0, 0, 255), 1, 8, 0)
    q2 = warped_img[256:350,344:707]

    st.image(details)
    snumber_from_filename = re.findall(r"-u[0-9]*", image.name)
    snumber_from_filename = re.sub("-", '', snumber_from_filename[0])
    snumber = st.text_input("Student Number", value = snumber_from_filename)

    row_index = df.index[df["Username"] == snumber_from_filename].tolist()
    surname = df.iloc[row_index, 0].values[0]
    first = df.iloc[row_index, 1].values[0]

    surname_entry = st.text_input("Surname", surname)
    first_entry = st.text_input("First Name", first)

    st.image(q1)
    chk1a = st.checkbox("Point A latitude", key="chk1a")
    chk1b = st.checkbox("Point A longitude", key="chk1b")
    chk1c = st.checkbox("Point B latitude", key="chk1c")
    chk1d = st.checkbox("Point B longitude", key="chk1d")
    chk1e = st.checkbox("Point C latitude", key="chk1e")
    chk1f = st.checkbox("Point C longitude", key="chk1f")

    st.image(q2)
    chk2a = st.checkbox("Point F latitude", key="chk2a")
    chk2b = st.checkbox("Point F longitude", key="chk2b")
    chk2c = st.checkbox("Point G latitude", key="chk2c")
    chk2d = st.checkbox("Point G longitude", key="chk2d")
    chk2e = st.checkbox("Point H latitude", key="chk2e")
    chk2f = st.checkbox("Point H longitude", key="chk2f")

    q1_grade = 0
    q2_grade = 0
    if st.button("Grade"):
        if st.session_state.chk1a:
            q1_grade += 1
        if st.session_state.chk1b:
            q1_grade += 1
        if st.session_state.chk1c:
            q1_grade += 1
        if st.session_state.chk1d:
            q1_grade += 1
        if st.session_state.chk1e:
            q1_grade += 1
        if st.session_state.chk1f:
            q1_grade += 1

        if st.session_state.chk2a:
            q2_grade += 1
        if st.session_state.chk2b:
            q2_grade += 1
        if st.session_state.chk2c:
            q2_grade += 1
        if st.session_state.chk2d:
            q2_grade += 1
        if st.session_state.chk2e:
            q2_grade += 1
        if st.session_state.chk2e:
            q2_grade += 1

        st.text_input("Q1 Grade", value=q1_grade)
        st.text_input("Q2 Grade", value=q2_grade)

        final_grade = q1_grade + q2_grade
        final_img = cv.putText(img=warped_img, text=f'{q1_grade}', org=(650, 380), fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        final_img = cv.putText(img=final_img, text=f'{q2_grade}', org=(650, 640),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        final_img = cv.putText(img=final_img, text=f'{final_grade}', org=(650, 150),
                               fontFace=cv.FONT_HERSHEY_SIMPLEX,
                               fontScale=1, color=(0, 0, 255), thickness=2)
        st.image(final_img)
        filename = f"{surname}-{first}.png"
        final_img_rgb = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)
        final_img_pil = Image.fromarray(final_img)
        buffer = io.BytesIO()
        final_img_pil.save(buffer, format="PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/jpeg")
