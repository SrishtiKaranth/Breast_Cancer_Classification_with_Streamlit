import numpy as np
import pickle 
import streamlit as st
import requests
import json
import streamlit_lottie
from streamlit_lottie import st_lottie

loaded_model = pickle.load(open("trained_model.sav", 'rb'))
loaded_scaler = pickle.load(open("scaler.pkl", 'rb'))

#creating a function 

def breast_cancer_prediction(single_obs):
    input_data_as_numpy_array=np.asarray(single_obs)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    standardized_input = loaded_scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(standardized_input)
    if(prediction==1):
       return "Cancer is Malignant"
    else:
        return "Cancer is Bening"
    

def main():

    st.set_page_config(page_title="Breast Cancer Classification",page_icon="☢",layout="wide")
    with st.container():
        left_column,right_column=st.columns([1,3])
        with left_column:
            st.title("Breast Cancer Classification")
            st.write("Enter the following details to help us classify the tumour as Malignant or Bening.🩺")
            st.write("##")
            st.write("##")

            #getting input data from user

            radius_mean=st.text_input('Radius mean of the tumour')
            texture_mean=st.text_input('Texture mean of the tumour')
            perimeter_mean=st.text_input('Perimeter mean of the tumour')
            area_mean=st.text_input('Area mean of the tumour')
            smoothness_mean=st.text_input('Smoothness mean of the tumour')
            compactness_mean=st.text_input('Compactness mean of the tumour')
            concavity_mean=st.text_input('Concavity mean of the tumour')
            concave_points_mean=st.text_input('Concave points mean of the tumour')
            symmetry_mean=st.text_input('Symmetry mean of the tumour')
            fractal_dimension_mean=st.text_input('Fractal dimension mean of the tumour')
            radius_se=st.text_input('Standard error of Radius of the tumour')
            texture_se=st.text_input('Standard error of Texture of the tumour')
            perimeter_se=st.text_input('Standard error of Perimeter of the tumour')
            area_se=st.text_input('Standard error of Area of the tumour')
            smoothness_se=st.text_input('Standard error of Smoothness of the tumour')
            compactness_se=st.text_input('Standard error of Compactness of the tumour')
            concavity_se=st.text_input('Standard error of Concavity of the tumour')
            concave_points_se=st.text_input('Standard error of Concave points of the tumour')
            symmetry_se=st.text_input('Standard error of Symmetry of the tumour')
            fractal_dimension_se=st.text_input('Standard error of Fractal dimension of the tumour')
            radius_worst=st.text_input('Worst Radius of the tumour')
            texture_worst=st.text_input('Worst Texture of the tumour')
            perimeter_worst=st.text_input('Worst Perimeter of the tumour')
            area_worst=st.text_input('Worst Area of the tumour')
            smoothness_worst=st.text_input('Worst Smoothness of the tumour')
            compactness_worst=st.text_input('Worst Compactness of the tumour')
            concavity_worst=st.text_input('Worst Concavity of the tumour')
            concave_points_worst=st.text_input('Worst Concave points of the tumour')
            symmetry_worst=st.text_input('Worst Symmetry of the tumour')
            fractal_dimension_worst=st.text_input('Worst Fractal dimension of the tumour')

            diagnosis=''

            if st.button("Predict Results"):
                diagnosis=breast_cancer_prediction([radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst])

            st.success(diagnosis)
            st.write("This model uses Logistic Regression and has an accuracy of 98%.")
            st.write("#")
            st.write("---")
        with right_column:
            st.markdown(
                        """
                        <div style="text-align: justify; margin-left: 70px; color:pink; font-size: 84px">
                        HOPE. HEAL. FIGHT.
                        </div>
                        """,
                        unsafe_allow_html=True,
                                )

            st.title("&nbsp; &nbsp; &nbsp; &nbsp;  What is Breast Cancer? 💉")
            st.subheader("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Breast Cancer is the most common cancer in women in India.")
            st.markdown(
                        """
                        <div style="text-align: justify; margin-left: 70px; color:grey;">
                        27.7% of all new cancers detected in women in India in the year 2018 were breast cancers.Breast cancer is a disease in which abnormal breast cells grow out of control and form tumours. If left unchecked, the tumours can spread throughout the body and become fatal. Breast cancer cells begin inside the milk ducts and/or the milk-producing lobules of the breast. The earliest form (in situ) is not life-threatening. Cancer cells can spread into nearby breast tissue (invasion). This creates tumours that cause lumps or thickening.Invasive cancers can spread to nearby lymph nodes or other organs (metastasize). Metastasis can be fatal.
                        </div>
                        """,
                        unsafe_allow_html=True,
                                )


            st.markdown("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [Learn more about breast cancer here >](https://www.breastcancerindia.net/)")
            st.write('#')
            st.write("---")
            st.write('#')
            with st.container():
                left_column,right_column=st.columns(2)
                with left_column:
                    st.subheader("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp;Scope of the problem 🎗")
                    st.markdown(
                        """
                        <div style="text-align: justify; margin-left: 40px; color:grey;">
                        In 2020, there were 2.3 million women diagnosed with breast cancer and 685 000 deaths globally. As of the end of 2020, there were 7.8 million women alive who were diagnosed with breast cancer in the past 5 years, making it the worlds most prevalent cancer. Breast cancer occurs in every country of the world in women at any age after puberty but with increasing rates in later life.  
                        </div>
                        """,
                        unsafe_allow_html=True,
                                )
                with right_column:
                    def load_lottiefile(filepath:str):
                        with open(filepath,"r") as f:
                            return json.load(f)
                    lottie_coding=load_lottiefile("animation_ln4fa54w.json")
                    st_lottie(
                    lottie_coding,
                    speed=1,
                    reverse=False,
                    loop=True,
                    quality="high",
                    # renderer="svg",
                    height="200px",
                    width="300px",
                    key=None,
                    )
    
            st.write('#')
            st.write("---")
            st.write('#')
            with st.container():
                left_column,right_column=st.columns([1,2])
                with right_column:
                    st.subheader("Types of Breast Cancer 🛑")
                    st.markdown(
                        """
                        <div style="text-align: justify; color:grey;">
                        Breast cancer encompasses a range of distinct types, each characterized by unique characteristics and behaviors. The most common form is Invasive Ductal Carcinoma (IDC), which begins in the milk ducts, spreads to nearby tissues, and can potentially metastasize to other parts of the body. On the other hand, Invasive Lobular Carcinoma (ILC) originates in the lobules and may not always form a distinct lump, making it harder to detect. Triple-Negative Breast Cancer lacks estrogen receptors, progesterone receptors, and HER2/neu receptors, posing challenges in treatment options. Conversely, HER2-Positive Breast Cancer involves overexpression of the HER2/neu gene and may respond well to targeted therapies. Additionally, Ductal Carcinoma In Situ (DCIS) is considered non-invasive as it remains within the milk ducts, while Inflammatory Breast Cancer is a rare, aggressive type characterized by redness, swelling, and warmth in the breast. Understanding these various breast cancer types is essential for tailored treatment and prognosis.  
                        </div>
                        """,
                        unsafe_allow_html=True,
                                )
                with left_column:
                    def load_lottiefile(filepath:str):
                        with open(filepath,"r") as f:
                            return json.load(f)
                    lottie_coding=load_lottiefile("animation_ln4q96ku.json")
                    st_lottie(
                    lottie_coding,
                    speed=1,
                    reverse=False,
                    loop=True,
                    quality="high",
                    # renderer="svg",
                    height="400px",
                    width="300px",
                    key=None,
                    )
    
            st.write('#')
            st.write("---")
            with st.container():
                left_column,right_column=st.columns(2)
                with left_column:
                    st.subheader("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; Signs and Symptoms")
                    st.markdown(
                        """
                        <div style="text-align: justify; margin-left: 40px; color:grey;">
                        Recognizing breast cancer signs is crucial for early detection. These signs may include discovering a new lump, noticing changes in breast size or shape, experiencing unexplained pain or tenderness, observing skin changes like redness or dimpling, and encountering nipple abnormalities such as inversion, discharge, or scaling. Additionally, be vigilant about persistent changes in breast texture or the appearance of swollen lymph nodes in the underarm area. While these symptoms don't confirm cancer, consulting a healthcare provider for a thorough evaluation is essential. Regular breast self-exams and mammograms are key components of maintaining breast health and facilitating early detection. 
                        </div>
                        """,
                        unsafe_allow_html=True,
                                )
                with right_column:
                    def load_lottiefile(filepath:str):
                        with open(filepath,"r") as f:
                            return json.load(f)
                    lottie_coding=load_lottiefile("animation_ln4pb6o3.json")
                    st_lottie(
                    lottie_coding,
                    speed=1,
                    reverse=False,
                    loop=True,
                    quality="high",
                    # renderer="svg",
                    height="350px",
                    width="500px",
                    key=None,
                    )
    
            st.write('#')
            st.write("---")
            st.write('#')
            with st.container():
                left_column,right_column=st.columns([1,2])
                with right_column:
                    st.subheader("Diagnosis 🩺")
                    st.markdown(
                        """
                        <div style="text-align: justify; color:grey;">
                        The diagnosis of breast cancer involves several steps. It starts with a clinical breast examination, followed by mammography. If necessary, further tests like ultrasound and MRI may be conducted. A definitive diagnosis is often made through a breast biopsy, where a small tissue sample is analyzed in a lab.

                        Once diagnosed, additional tests determine the extent of the disease, including its stage. These diagnostic steps are crucial for developing a personalized treatment plan tailored to the specific characteristics of the breast cancer. Early detection and accurate diagnosis are essential for better outcomes.  
                        </div>
                        """,
                        unsafe_allow_html=True,
                                )
                with left_column:
                    def load_lottiefile(filepath:str):
                        with open(filepath,"r") as f:
                            return json.load(f)
                    lottie_coding=load_lottiefile("animation_ln4pj1t1.json")
                    st_lottie(
                    lottie_coding,
                    speed=2,
                    reverse=False,
                    loop=True,
                    quality="high",
                    # renderer="svg",
                    height="300px",
                    width="300px",
                    key=None,
                    )

            st.write("---")
            with st.container():
                left_column,right_column=st.columns(2)
                with left_column:
                    st.subheader("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Treatment Options")
                    st.markdown(
                        """
                        <div style="text-align: justify; margin-left: 40px; color:grey;">
                        Breast cancer treatment options include surgery, radiation therapy, chemotherapy, targeted therapy, hormone therapy, and immunotherapy. These treatments are tailored based on factors like cancer stage and type. Combination approaches are often used for the best outcomes. 
                        </div>
                        """,
                        unsafe_allow_html=True,
                                )
                with right_column:
                    def load_lottiefile(filepath:str):
                        with open(filepath,"r") as f:
                            return json.load(f)
                    lottie_coding=load_lottiefile("animation_ln4ps1od.json")
                    st_lottie(
                    lottie_coding,
                    speed=1,
                    reverse=False,
                    loop=True,
                    quality="high",
                    # renderer="svg",
                    height="200px",
                    width="300px",
                    key=None,
                    )
            st.write("---")
            st.write('#')
            with st.container():
                left_column,right_column=st.columns([1,2])
                with right_column:
                    st.subheader("Fighting Cancer 💪")
                    st.markdown(
                        """
                        <div style="text-align: justify; color:grey;">
                        Overcoming breast cancer requires strength, support, and a tailored treatment plan. Emotional and psychological support, along with a strong network of friends and family, play crucial roles. Staying informed, joining support groups, and maintaining a hopeful outlook contribute to success. Survivors serve as inspiring examples of resilience and triumph over adversity.  
                        </div>
                        """,
                        unsafe_allow_html=True,
                                )
                with left_column:
                    def load_lottiefile(filepath:str):
                        with open(filepath,"r") as f:
                            return json.load(f)
                    lottie_coding=load_lottiefile("animation_ln4qc4mk.json")
                    st_lottie(
                    lottie_coding,
                    speed=1,
                    reverse=False,
                    loop=True,
                    quality="high",
                    # renderer="svg",
                    height="200px",
                    width="300px",
                    key=None,
                    )
            st.write('#')
            





if __name__ == '__main__':
    main()