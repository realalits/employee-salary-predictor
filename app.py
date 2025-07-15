# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import joblib

# # st.title("Employee Salary Prediction")
# # st.divider()
# # st.write("This app predicts the salary of employees based on various features.")
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load dataset (for dropdown options or validation)
# df = pd.read_csv(r"E:\VS CODE\python\EmpSalaryPridiction\filterdata_employee.csv")

# # Load trained model
# model = joblib.load(r"E:\VS CODE\python\EmpSalaryPridiction\salary_model.joblib")

# # Streamlit UI
# st.title("Employee Salary Prediction")
# st.divider()
# st.write("This app predicts the salary of employees based on their profile.")

# # Extract possible values for dropdowns (assuming these columns exist)
# education_levels = sorted(df["Education Level"].dropna().unique())
# locations = sorted(df["Location"].dropna().unique())

# # Collect user input
# st.header("Enter Employee Details")

# experience = st.slider("Years of Experience", 0, 40, 5)
# education = st.selectbox("Education Level", education_levels)
# location = st.selectbox("Location", locations)

# # Convert to DataFrame for model prediction
# input_df = pd.DataFrame({
#     "Years of Experience": [experience],
#     "Education Level": [education],
#     "Location": [location]
# })

# # Predict and display
# if st.button("Predict Salary"):
#     prediction = model.predict(input_df)[0]
#     st.success(f"Predicted Salary: ${prediction:,.2f}")



import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load data
filter_df = pd.read_csv("filterdata_employee.csv")
number_df = pd.read_csv("numberdata_employee.csv")
model = joblib.load("salary_model.joblib")

st.title("Employee Salary Prediction")
st.divider()
st.write("This app predicts the salary of employees based on their profile.")

# Generate mappings from filter_df to number_df
mappings = {}
for col in filter_df.columns:
    if filter_df[col].dtype == "object":
        unique_vals = filter_df[col].dropna().unique()
        map_vals = {
            val: number_df[filter_df[col] == val][col].iloc[0]
            for val in unique_vals
        }
        mappings[col] = map_vals

# User inputs
age = st.slider("Age", 18, 70, 30)
workclass = st.selectbox("Workclass", sorted(filter_df["workclass"].dropna().unique()))
education = st.selectbox("Education", sorted(filter_df["education"].dropna().unique()))
marital_status = st.selectbox("Marital Status", sorted(filter_df["marital-status"].dropna().unique()))
occupation = st.selectbox("Occupation", sorted(filter_df["occupation"].dropna().unique()))
relationship = st.selectbox("Relationship", sorted(filter_df["relationship"].dropna().unique()))
race = st.selectbox("Race", sorted(filter_df["race"].dropna().unique()))
gender = st.selectbox("Gender", sorted(filter_df["gender"].dropna().unique()))
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
native_country = st.selectbox("Native Country", sorted(filter_df["native-country"].dropna().unique()))
fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=0, value=100000)
educational_num = st.slider("Educational Number", 1, 16, 10)

# Prepare input as DataFrame
input_dict = {
    "age": age,
    "workclass": mappings["workclass"][workclass],
    "fnlwgt": fnlwgt,
    "education": mappings["education"][education],
    "educational-num": educational_num,
    "marital-status": mappings["marital-status"][marital_status],
    "occupation": mappings["occupation"][occupation],
    "relationship": mappings["relationship"][relationship],
    "race": mappings["race"][race],
    "gender": mappings["gender"][gender],
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week,
    "native-country": mappings["native-country"][native_country]
}

input_df = pd.DataFrame([input_dict])

# Predict and display
if st.button("Predict Salary"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Salary: {prediction}")



    
# # import streamlit as st
# # import pandas as pd 
# # import numpy as np
# # import joblib
# # st.title("Employee Salary Prediction")
# # st.divider()
# # st.write("This app predicts the salary of employees based on various features.")
# # df = pd.read_csv(r"E:\VS CODE\python\EmpSalaryPridiction\filterdata_employee.csv")
# # model = joblib.load(r"E:\VS CODE\python\EmpSalaryPridiction\salary_model.joblib")
# # st.title("Employee Salary Prediction")
# # st.divider()
# # st.write("This app predicts the salary of employees based on their profile.")
# # # Extract possible values for dropdowns (assuming these columns exist)
# # education_levels = sorted(df["Education Level"].dropna().unique())  





# # locations = sorted(df["Location"].dropna().unique())
# # # Collect user input
# # st.header("Enter Employee Details")
# # experience = st.slider("Years of Experience", 0, 40, 5)
# # education = st.selectbox("Education Level", education_levels)   
# # location = st.selectbox("Location", locations)
# # # Convert to DataFrame for model prediction
# # input_df = pd.DataFrame({
#     "Years of Experience": [experience],
#     "Education Level": [education],
#     "Location": [location]
# })
# # Predict and display
# # if st.button("Predict Salary"):
#     prediction = model.predict(input_df)[0]
#     st.success(f"Predicted Salary: ${prediction:,.2f}")
# # import streamlit as st
# # import pandas as pd
# # import numpy as np  
# # import joblib
# # st.title("Employee Salary Prediction")
# # st.divider()
# # st.write("This app predicts the salary of employees based on various features.")
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import qgrid

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder  
# from sklearn.ensemble import RandomForestClassifier