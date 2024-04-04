import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
from stqdm import stqdm

# Title Tab
st.set_page_config(page_title="Prediction", page_icon="favicon_32x32_XX0_icon.ico")

# Header
st.header("🤖 Sales Predictor from Ad Spending 💸")
st.subheader('', divider='rainbow')

# Data and DataFrame
url = 'https://raw.githubusercontent.com/modpanyakorn/prediction-data-using-linear-regression-model/main/Advertising.csv'
df = pd.read_csv(url)
df = df.drop(['Unnamed: 0'], axis=1)
st.write("ข้อมูล Ad Spending")
st.write(df)

# Sidebar
st.sidebar.markdown("<h1 style='text-align: center;'>Predictor</h1>", unsafe_allow_html=True)
st.sidebar.subheader('', divider='rainbow')
X = st.sidebar.selectbox("เลือกข้อมูลสำหรับ Training และ Scatter Plot", df[['TV', 'Radio', 'Newspaper']].columns)
train = st.sidebar.selectbox("เลือกวิธีการประมาณค่าพารามิเตอร์ (Training)", ["Least Squares Approximation", "Gradient Descent"])

# Scatter Plot2D
st.subheader('', divider='rainbow')
fig2D = go.Figure()
if X == "TV":
    st.subheader(f"Scatter Plot ระหว่าง {X} กับ Sales")
    fig2D.add_trace(go.Scatter(x=df['TV'], y=df['Sales'], mode='markers', name='Data Point'))
    fig2D.update_layout(
        xaxis_title=X + " (X)",
        yaxis_title="Sales (y)"
    )
    st.plotly_chart(fig2D)
    
elif X == "Radio":
    st.subheader(f"Scatter Plot ระหว่าง {X} กับ Sales")
    fig2D.add_trace(go.Scatter(x=df['Radio'], y=df['Sales'], mode='markers', name='Data Point'))
    fig2D.update_layout(
        xaxis_title=X + " (X)",
        yaxis_title="Sales (y)"
    )
    st.plotly_chart(fig2D)

elif X == "Newspaper":
    st.subheader(f"Scatter Plot ระหว่าง {X} กับ Sales")
    fig2D.add_trace(go.Scatter(x=df['Newspaper'], y=df['Sales'], mode='markers', name='Data Point'))
    fig2D.update_layout(
        xaxis_title=X + " (X)",
        yaxis_title="Sales (y)"
    )
    st.plotly_chart(fig2D)

# Approximation Parameters
st.write(train)
if train == "Least Squares Approximation":
    # btn train
    button = st.sidebar.button("Training", type="primary")
    # preprocess
    X = df[X].values
    ones = np.ones(len(X))
    X = np.stack((ones, X)).T
    y = df[['Sales']].values
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # approximation formula 
    if button:
        Xtran_dot_X = X.T @ X
        Xtran_dot_y = X.T @ y
        beta = np.linalg.inv(Xtran_dot_X) @ (Xtran_dot_y)
        # parameters
        beta_0, beta_1 = beta[0, 0], beta[1, 0]
        # st.write(beta_0, beta_1)

elif train == "Gradient Descent":
    # add text-box in sidebar
    iteration = int(st.sidebar.number_input("จำนวนรอบการเรียนรู้ (Training)", min_value=0, value=1000000))
    print(iteration)
    learning_rate = float(st.sidebar.text_input("จำนวนอัตตราการเรียนรู้ (Learning Rate)", value=1e-7))
    print(type(learning_rate), learning_rate)
    # preprocess
    X = np.array(df[X])
    y = np.array(df['Sales'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # brn train
    button = st.sidebar.button("Training", type="primary")
    # declare parameters vars
    beta_0 = 0
    beta_1 = 0
    # declare sse var
    sse_list = []
    if button:
    # GD Optimization Algorithm
        round_epochs = 10000
        with stqdm(total = iteration) as pbar:
            for i in range(iteration):
                sse = np.sum((y - beta_0 - beta_1 * X) ** 2)
                diff_sse_beta_0 = np.sum(-2 * (y - beta_0 - beta_1 * X))
                diff_sse_beta_1 = np.sum(-2 * (y - beta_0 - beta_1 * X) * X)

                step_size_beta_0 = diff_sse_beta_0 * learning_rate
                step_size_beta_1 = diff_sse_beta_1 * learning_rate

                beta_0 = beta_0 - step_size_beta_0
                beta_1 = beta_1 - step_size_beta_1

                sse_list.append(sse)

                if (i + 1) % round_epochs == 0:
                    pbar.set_description(f'Iterate: {i + 1}, SSE: {sse}')
                    pbar.update(round_epochs)
        approx_done = st.success(f"Intercept: {beta_0}, Slope: {beta_1}")

# print(beta_0, beta_1)