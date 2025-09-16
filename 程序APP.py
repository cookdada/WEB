import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('RF.pkl')

# 优化后的特征范围定义
feature_ranges = {
    "EFW": {"type": "numerical", "min": 2289.0, "max": 4108.0, "default": 3000.0},
    "Platelet": {"type": "numerical", "min": 105.0, "max": 394.0, "default": 250.0},
    "ALT": {"type": "numerical", "min": 2.0, "max": 179.0, "default": 30.0},
    "Macrosomia": {"type": "categorical", "options": [0, 1]},
    "Bishop": {"type": "numerical", "min": 1.0, "max": 6.0, "default": 3.0},
    "Parity": {"type": "numerical", "min": 0.0, "max": 3.0, "default": 1.0},
    "Cervical effacement": {"type": "numerical", "min": 0.0, "max": 3.0, "default": 1.5},
    "Hgb": {"type": "numerical", "min": 79.0, "max": 145.0, "default": 120.0},
    "Gravidity": {"type": "numerical", "min": 1.0, "max": 6.0, "default": 3.0},
    "BMI": {"type": "numerical", "min": 18.5, "max": 36.9, "default": 25.0},
    "Gestational weight gain": {"type": "numerical", "min": 0.5, "max": 29.6, "default": 15.0},
    "Fetal station": {"type": "numerical", "min": 0.0, "max": 1.0, "default": 0},
}

# Streamlit 界面
st.title("Prediction Model")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测结果显示
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of emergency cesarean section is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")



