import numpy as np
import pickle
import streamlit as st
import pandas as pd


# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))


REQ_COLS = [
    'satisfaction_level', 'last_evaluation', 'number_project',
    'average_montly_hours', 'time_spend_company', 'Work_accident',
    'promotion_last_5years',
    'Department_IT', 'Department_RandD', 'Department_accounting',
    'Department_hr', 'Department_management', 'Department_marketing',
    'Department_product_mng', 'Department_sales', 'Department_support',
    'Department_technical', 'salary_high', 'salary_low', 'salary_medium'
]


# creating a function for Prediction

def employee_exit_prediction(input_data):

    satisfaction_level, last_evaluation, number_project, \
    average_montly_hours, time_spend_company, Work_accident, \
    promotion_last_5years, Department, salary = input_data

    df = pd.DataFrame([{
        'satisfaction_level':    float(satisfaction_level),
        'last_evaluation':       float(last_evaluation),
        'number_project':        int(number_project),
        'average_montly_hours':  int(average_montly_hours),
        'time_spend_company':    int(time_spend_company),
        'Work_accident':         int(Work_accident),
        'promotion_last_5years': int(promotion_last_5years),
        'Department':            Department,
        'salary':                salary,
    }])

    df = pd.get_dummies(df, columns=['Department', 'salary'])

    for col in REQ_COLS:
        if col not in df.columns:
            df[col] = 0

    df = df[REQ_COLS]
    df_scaled = scaler.transform(df)

    prediction = loaded_model.predict(df_scaled)

    if prediction[0] == 1:
        return 'The Employee will Exit the Company'
    else:
        return 'The Employee will Stay in the Company'


def main():

    # giving a title
    st.title('Employee Exit Prediction Web App')

    # getting the input data from the user
    satisfaction_level    = st.text_input('Satisfaction Level (0.0 - 1.0)')
    last_evaluation       = st.text_input('Last Evaluation Score (0.0 - 1.0)')
    number_project        = st.text_input('Number of Projects')
    average_montly_hours  = st.text_input('Average Monthly Hours')
    time_spend_company    = st.text_input('Years Spent in Company')
    Work_accident         = st.text_input('Work Accident (0 = No, 1 = Yes)')
    promotion_last_5years = st.text_input('Promotion in Last 5 Years (0 = No, 1 = Yes)')
    Department            = st.text_input('Department (sales / IT / hr / accounting / support / technical / management / marketing / RandD / product_mng)')
    salary                = st.text_input('Salary Level (low / medium / high)')

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Employee Exit Test Result'):
        diagnosis = employee_exit_prediction([
            satisfaction_level, last_evaluation, number_project,
            average_montly_hours, time_spend_company, Work_accident,
            promotion_last_5years, Department, salary
        ])

    st.success(diagnosis)


if __name__ == '__main__':
    main()