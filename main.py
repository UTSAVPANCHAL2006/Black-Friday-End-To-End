import streamlit as st
import requests

st.set_page_config(page_title="User Purchase Prediction", layout="centered")

st.title("User Purchase Prediction App")
st.write("Enter the user and product details to predict the purchase amount.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["F", "M"])
        age = st.selectbox("Age", ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"])
        occupation = st.number_input("Occupation", min_value=0, step=1, value=0)
        city_category = st.selectbox("City Category", ["A", "B", "C"])
        stay_years = st.selectbox("Stay In Current City Years", ["0", "1", "2", "3", "4+"])
        marital_status = st.selectbox("Marital Status", [0, 1])
        
    with col2:
        product_id = st.text_input("Product ID", "P00069042")
        product_category_1 = st.number_input("Product Category 1", min_value=1, max_value=20, value=1)
        product_category_2 = st.number_input("Product Category 2", min_value=0.0, max_value=20.0, value=0.0)
        product_category_3 = st.number_input("Product Category 3", min_value=0.0, max_value=20.0, value=0.0)

    submit = st.form_submit_button("Predict")

if submit:
    payload = {
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "City_Category": city_category,
        "Stay_In_Current_City_Years": stay_years,
        "Marital_Status": marital_status,
        "Product_Category_1": product_category_1,
        "Product_Category_2": product_category_2,
        "Product_Category_3": product_category_3,
        "Product_ID": product_id
    }
    
    with st.spinner("Predicting..."):
        try:

            response = requests.post("https://black-friday-end-to-end.onrender.com", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('prediction', 0)
                warning = result.get('warning', '')
                
                st.success(f"Predicted Purchase Amount: ${prediction:.2f}")
                
                if warning:
                    st.warning(warning)
            else:
                st.error(f"Error from API: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend. Please ensure the FastAPI server is running on http://127.0.0.1:8000.")
