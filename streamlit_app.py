import streamlit as st
import google.generativeai as genai
import requests
import base64
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
import re
import os

# ===== 配置：从 Streamlit Cloud 的 Secrets 中读取密钥 =====
BAIDU_API_KEY = st.secrets["BAIDU_API_KEY"]
BAIDU_SECRET_KEY = st.secrets["BAIDU_SECRET_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# 配置 Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')  # 使用 gemini-pro 文本模型

# ===== 百度 OCR 函数（保持不变） =====
def baidu_ocr(image_bytes):
    """调用百度通用文字识别（高精度版）"""
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    # 获取 access_token
    token_host = "https://aip.baidubce.com/oauth/2.0/token"
    token_params = {
        "grant_type": "client_credentials",
        "client_id": BAIDU_API_KEY,
        "client_secret": BAIDU_SECRET_KEY
    }
    token_res = requests.get(token_host, params=token_params)
    access_token = token_res.json().get("access_token")

    # 发送 OCR 请求
    img_base64 = base64.b64encode(image_bytes).decode()
    ocr_url = url + f"?access_token={access_token}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"image": img_base64}
    response = requests.post(ocr_url, headers=headers, data=data)
    result = response.json()

    # 提取文字
    words = []
    if "words_result" in result:
        for item in result["words_result"]:
            words.append(item["words"])
    return "\n".join(words)

# ===== 调用 Gemini 生成教学反馈 =====
def generate_feedback(text_input, class_level, question_count):
    """
    使用 Gemini 生成反馈：
    - text_input: 从图片识别出的文字
    - class_level: 年级（如"三年级"）
    - question_count: 题目数量
    """
    # 备用模板（当 Gemini 不可用时）
    fallback_templates = {
        "三年级": "整体完成得很好！计算方面可以再细心一点，应用题思路清晰。",
        "四年级": "解题步骤规范，能够正确运用公式，继续保持！",
        "五年级": "对于稍复杂的题目，尝试画图辅助理解，正确率很高！",
        "六年级": "已经掌握了六年级的核心知识点，审题时注意单位统一。"
    }

    try:
        # 构建 prompt
        prompt = f"""你是一位经验丰富的{class_level}数学老师，请根据学生作业图片中识别出的文字内容，生成一段面向家长和学生的反馈。
反馈要求：
1. 语言亲切、鼓励，指出优点和需改进的地方。
2. 针对{question_count}道题的完成情况。
3. 如果识别内容不完整，根据常见题目类型合理补充。
4. 反馈长度约80-150字。

识别出的文字：
{text_input if text_input else "（未识别到文字，可能是纯手写或图片不清晰）"}
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Gemini 调用失败，使用备用模板
        fb = fallback_templates.get(class_level, "作业完成认真，继续努力！")
        st.warning(f"Gemini 暂时不可用，显示默认反馈：{fb}")
        return fb

# ===== 页面布局 =====
st.set_page_config(page_title="数学教学助手", layout="wide")
st.title("📐 数学作业智能反馈助手")
st.markdown("上传学生作业图片，自动识别题目并生成个性化反馈。")

# 侧边栏选择设置
with st.sidebar:
    st.header("⚙️ 设置")
    class_level = st.selectbox("年级", ["三年级", "四年级", "五年级", "六年级"])
    question_count = st.slider("本次作业题数", 1, 10, 5)

# 主区域：上传图片
uploaded_file = st.file_uploader("上传作业图片（支持 JPG/PNG）", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示原图
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 上传的图片")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    # 保存为字节流供 OCR 使用
    img_bytes = uploaded_file.getvalue()

    with col2:
        st.subheader("🔍 识别结果")
        with st.spinner("正在识别文字..."):
            recognized_text = baidu_ocr(img_bytes)
        if recognized_text:
            st.text_area("识别出的文字", recognized_text, height=150)
        else:
            st.warning("未识别到文字，可能图片模糊或为纯手写，将继续生成反馈。")
            recognized_text = ""

    # 生成反馈按钮
    if st.button("🚀 生成教学反馈"):
        with st.spinner("Gemini 正在思考反馈内容..."):
            feedback = generate_feedback(recognized_text, class_level, question_count)
        st.subheader("📝 教师反馈（可复制到班级群或打印）")
        st.success(feedback)

        # 同时显示原始识别文本，方便核对
        with st.expander("查看原始识别数据"):
            st.text(recognized_text)
