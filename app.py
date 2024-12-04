import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from recomand import *
def get_image_base64(img_path):
    with open(img_path, "rb") as img_file:
        img_bytes = img_file.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64
def home_page():
    add_bg_from_local('back3.jpg')
    st.markdown("""  
        <div style='text-align: center; padding: 20px; 
                    background: rgba(255, 255, 255, 0.7); 
                    margin-bottom: 40px; width: 100%;'>
            <h1 style='color: #2c3e50;'>Brain Tumor Detection Using MRI</h1>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 3], gap="large")  

    with col1:
        img_base64 = get_image_base64("back.jpg")
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{img_base64}" alt="Home Image" 
                     style="width: 100%; max-width: 100%; height: 400px; border-radius: 10px;">
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            '''
            <div style='text-align: justify; color:black; font-size: 22px; width: 100%;'>
               Our brain tumor detection web application leverages advanced machine learning algorithms to analyze MRI (Magnetic Resonance Imaging) images for the early identification of brain tumors. Users can easily upload their MRI images in formats like JPG and PNG, receiving immediate feedback on any detected tumors and their types. The app highlights tumor areas for better understanding and includes educational resources on symptoms and treatment options. With a focus on patient privacy and data security, our app aims to enhance diagnostic efficiency and improve patient outcomes through accessible healthcare management.
            </div>
            ''',
            unsafe_allow_html=True
        )
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.4)), 
            url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
def prediction_page():    
    add_bg_from_local("back3.jpg")    
    st.markdown("""  
            <div style='text-align: center; padding: 20px; 
                        background: rgba(255, 255, 255, 0.7); 
                        margin-bottom: 40px; width: 100%;'>
                <h1 style='color: #2c3e50;'>Brain Tumor Detection Using MRI</h1>
            </div>
        """, unsafe_allow_html=True)
    cnn = tf.keras.models.load_model('trained_brain_tumor.keras')
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    col1, col2 = st.columns([1, 1], gap="large") 
    with col1:
        st.markdown(f"<h3 style='color: black;'>Upload an MRI image to predict if there is a tumor and which type:</h3>", unsafe_allow_html=True)
    #st.write("Upload an MRI image to predict if there is a tumor and which type.")

    with col2:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        #st.image(image, caption="Uploaded MRI image", use_column_width=True)
        
        image = image.resize((128, 128))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        
        predictions = cnn.predict(input_arr)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]
        
        #st.write(f"Prediction: **{predicted_class_name}**")
        #st.markdown(f"<h3 style='color: yellow;'>Prediction : {predicted_class_name}</h3>", unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2], gap="large")  

        with col1:
            st.markdown(f"<h3 style='color: black;'>Uploaded MRI Image:</h3>", unsafe_allow_html=True)
            st.image(image, caption="", use_column_width=True)

        with col2:
                st.markdown(f"<h3 style='color: purple;'>Prediction : {predicted_class_name}</h3>", unsafe_allow_html=True)
                if st.button("Get Information"):
                    recommandation=give_recommandation(predicted_class_name)
                    html_content = f"""
                                            <div style='color: black; text-align: justify; font-size:20px; padding-left:10px;'>
                                                <div style='margin-top: 10px; padding: 15px; border: 2px solid black; 
                                                border-radius: 8px; background-color: rgba(255, 255, 255, 0.9);'>
                                                    {recommandation.replace('\n', '<br>')}
                                                </div>
                                            </div>
                                            """
            
                                            # Display the HTML content
                    st.markdown(html_content, unsafe_allow_html=True)

def about_us_page():
    add_bg_from_local('back5.jpg')
    st.markdown("<h2 style='text-align: center;color:black;'>About Us</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 18px;color:black;'>
        Welcome to Brain Tumor Detection WebApp, a platform designed to aid in the fast and accurate detection of brain tumors using advanced deep learning models. Our goal is to support early diagnosis and improve treatment outcomes by providing reliable, easy-to-use tools for medical professionals.
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="wide")
    add_bg_from_local("back4.jpg")
    st.sidebar.title("Navigation")

    st.markdown(
    """
    <style>
    /* Sidebar background color with transparency and fixed width */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.5) !important;
        color: white !important;
        width: 200px !important;  /* Set fixed width */
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] .css-1v3fvcr {
        color: white !important;
    }

    /* Remove default margin padding */
    .css-18e3th9 {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Center align the sidebar content */
    [data-testid="stSidebar"] .css-1aumxhk {
        text-align: center;
    }

    /* Adjust the main content padding */
    .css-1d391kg {
        padding: 1rem 3rem 1rem 1rem;
    }

    /* Adjust width for the sidebar content */
    [data-testid="stSidebar"] .css-1lcbmhc {
        width: 200px !important;
    }
    
    </style>
    """,
    unsafe_allow_html=True
    )
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    if st.sidebar.button("Home"):
        st.session_state['page'] = "Home"
    if st.sidebar.button("Tumor Prediction"):
        st.session_state['page'] = "Tumor Prediction"
    if st.sidebar.button("About Us"):
        st.session_state['page'] = "About Us"

    if st.session_state['page'] == "Home":
        home_page()
    elif st.session_state['page'] == "Tumor Prediction":
        prediction_page()    
    elif st.session_state['page'] == "About Us":
        about_us_page()

if __name__ == "__main__":
    main()
