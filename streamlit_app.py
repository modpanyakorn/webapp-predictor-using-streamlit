import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
st.sidebar.markdown("<h1 style='text-align: center;'>Approximate Parameters</h1>", unsafe_allow_html=True)
st.sidebar.subheader('', divider='rainbow')
X = st.sidebar.selectbox("เลือกข้อมูลสำหรับ Training และ Scatter Plot", df[['TV', 'Radio', 'Newspaper']].columns)
train = st.sidebar.selectbox("เลือกวิธีการประมาณค่าพารามิเตอร์ (Training)", ["None", "Least Squares Approximation", "Gradient Descent"])
button_train = st.sidebar.button("Training", type="primary")

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
#  and button_train == False and train not in ["Least Squares Approximation", "Gradient Descent"]
print(button_train, train)
if ['st.session_state.beta_0', 'st.session_state.beta_1', 'st.session_state.beta_0GD', 'st.session_state.beta_1GD'] not in st.session_state and button_train != True and train not in ["Gradient Descent", "Least Squares Approximation"]:
    st.session_state.beta_0 = 0
    st.session_state.beta_1 = 0 
    st.session_state.beta_0GD = 0
    st.session_state.beta_1GD = 0
    print("if")
elif train == "Least Squares Approximation" and button_train == True:
    st.session_state.display = True
    # preprocess
    X_pre = df[X].values
    ones = np.ones(len(X_pre))
    X_pre = np.stack((ones, X_pre)).T
    y = df[['Sales']].values
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X_pre, y, test_size=0.25, random_state=42)
    # approximation formula 
    Xtran_dot_X = X_train.T @ X_train
    Xtran_dot_y = X_train.T @ y_train
    beta = np.linalg.inv(Xtran_dot_X) @ (Xtran_dot_y)
    # parameters
    st.session_state.beta_0, st.session_state.beta_1 = beta[0, 0], beta[1, 0]
    
    st.info(f"Intercept: {st.session_state.beta_0}, Slope: {st.session_state.beta_1}")
    # r2
    y_pred_ls = X_test @ beta
    r2 = r2_score(y_test, y_pred_ls)
    st.success(f"R-squared: {round(r2, 5)} หรือ {round(r2 * 100)}%")
    # SSE
    sse = (y_train - X_train @ beta).T @ (y_train - X_train @ beta)
    st.error(f"SSE (Sum of Squares Error): {round(sse[0, 0], 5)}")
    # scatter plot
    fig2D_LS = go.Figure()
    fig2D_LS.add_trace(go.Scatter(x=X_train[:, 1], y=y_train[:, 0], mode='markers', name='Data Point'))
    fig2D_LS.add_trace(go.Scatter(x=X_train[:, 1], y=st.session_state.beta_0 + st.session_state.beta_1 * X_train[:, 1], mode='lines', name='Regression Line', line_color='red'))
    fig2D_LS.update_layout(
        xaxis_title=X + " (X)",
        yaxis_title="Sales (y)"
    )
    st.plotly_chart(fig2D_LS)

elif train == "Gradient Descent":
    st.session_state.display = True
    # add text-box in sidebar
    iteration = int(st.sidebar.number_input("จำนวนรอบการเรียนรู้ (Training)", min_value=0, value=1000000, step=10000))
    learning_rate = float(st.sidebar.text_input("จำนวนอัตตราการเรียนรู้ (Learning Rate)", value=1e-7))
    # preprocess
    X_split = np.array(df[X])
    y_split = np.array(df['Sales'])
    X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.25, random_state=42)
    # declare sse var
    beta_0, beta_1 = 0, 0
    sse_list = []
    # GD Optimization Algorithm
    round_epochs = 10000
    if button_train:
        with stqdm(total = iteration) as pbar:
            for i in range(iteration):
                sse = np.sum((y_train - beta_0 - beta_1 * X_train) ** 2)
                diff_sse_beta_0 = np.sum(-2 * (y_train - beta_0 - beta_1 * X_train))
                diff_sse_beta_1 = np.sum(-2 * (y_train - beta_0 - beta_1 * X_train) * X_train)

                step_size_beta_0 = diff_sse_beta_0 * learning_rate
                step_size_beta_1 = diff_sse_beta_1 * learning_rate

                beta_0 = beta_0 - step_size_beta_0
                
                beta_1 = beta_1 - step_size_beta_1
                sse_list.append(sse)

                if (i + 1) % round_epochs == 0:
                    pbar.set_description(f'Iterate: {i + 1}, SSE: {sse}')
                    pbar.update(round_epochs)
        st.session_state.beta_0GD = beta_0
        st.session_state.beta_1GD = beta_1
        st.info(f"Intercept: {beta_0}, Slope: {beta_1}")
        # r2
        y_pred_test_set = beta_0 + (beta_1 * X_test)
        r2 = r2_score(y_test, y_pred_test_set)
        st.success(f"R-squared: {round(r2, 5)} หรือ {round(r2 * 100)}%")
        # SSE
        st.error(f"SSE (Sum of Squares Error): {sse_list[-1]}")
        
        # scatter plot
        fig2D_GD = go.Figure()
        fig2D_GD.add_trace(go.Scatter(x=df[X], y=df['Sales'], mode='markers', name='Data Point'))
        fig2D_GD.add_trace(go.Scatter(x=df[X], y=beta_0 + beta_1 * df[X], mode='lines', name='Regression Line', line_color='red'))
        fig2D_GD.update_layout(
            xaxis_title=X + " (X)",
            yaxis_title="Sales (y)"
        )
        st.plotly_chart(fig2D_GD)

print(st.session_state.beta_0, st.session_state.beta_1, st.session_state.beta_0GD, st.session_state.beta_1GD)
# st.session_state.input_X = 0
# if st.session_state.input_X == 0:
#     st.error("for Predict Training First!!")
if train == "Least Squares Approximation" and st.session_state.beta_0 != 0:
    input_X = st.number_input(f'ทำนายยอดขาย Sales จากการลงทุนโฆษณาใน {X}')
    st.warning(f"ทำนายยอดขาย Sales ได้: {round(st.session_state.beta_0 + st.session_state.beta_1 * input_X, 5)}")
elif train == "Gradient Descent" and st.session_state.beta_0GD != 0:
    input_X = st.number_input(f'ทำนายยอดขาย Sales จากการลงทุนโฆษณาใน {X}')
    st.warning(f"ทำนายยอดขาย Sales ได้: {round(st.session_state.beta_0GD + st.session_state.beta_1GD * input_X, 5)}")