import streamlit as st
import joblib
import numpy as np

# 加载保存的逻辑回归模型
model = joblib.load('lr.pkl')

# 定义预测函数
def predict_delivery(Gestational_week, Parity, Cervical_position, Gravidity, Fetal_station, Nuchal_cord, Femur_length, Bishop, Fetal_abdominal_circumference, Cervical_effacement):
    # 将输入数据合并为一个数组
    input_data = np.array([[Gestational_week, Parity, Cervical_position, Gravidity, Fetal_station, Nuchal_cord, Femur_length, Bishop, Fetal_abdominal_circumference, Cervical_effacement]])
    
    # 使用模型进行预测
    prediction = model.predict(input_data)[0]
    
    # 获取概率
    probability = model.predict_proba(input_data)[0][1]
    
    # 输出预测结果和概率
    return f"Predicted Outcome: {'emergency cesarean section' if prediction == 1 else 'vaginal delivery'}", f"Probability: {probability:.2f}"

# 创建 Streamlit 应用
st.title("Delivery Outcome Prediction")

# 获取输入特征
Gestational_week = st.number_input("Gestational week", value=40)
Parity = st.number_input("Parity", value=0)
Cervical_position = st.number_input("Cervical position", value=1)
Gravidity = st.number_input("Gravidity", value=1)
Fetal_station = st.number_input("Fetal station", value=0)
Nuchal_cord = st.number_input("Nuchal cord", value=1)
Femur_length = st.number_input("Femur length", value=71)
Bishop = st.number_input("Bishop", value=3)
Fetal_abdominal_circumference = st.number_input("Fetal abdominal circumference", value=348)
Cervical_effacement = st.number_input("Cervical effacement", value=1)

# 创建按钮并显示预测结果
if st.button("Predict Outcome"):
    outcome, probability = predict_delivery(
        Gestational_week, Parity, Cervical_position, Gravidity, Fetal_station,
        Nuchal_cord, Femur_length, Bishop, Fetal_abdominal_circumference, Cervical_effacement
    )
    st.subheader(outcome)
    st.write(f"Probability: {probability}")
