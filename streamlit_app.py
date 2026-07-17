import streamlit as st
import requests
import base64
import json
import re
import io
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import tempfile
import os
import pdfplumber  # 处理 PDF
from docx import Document  # 处理 Word

# ===== 配置 =====
BAIDU_API_KEY = st.secrets["BAIDU_API_KEY"]
BAIDU_SECRET_KEY = st.secrets["BAIDU_SECRET_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_PROXY = st.secrets.get("GEMINI_PROXY", None)

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

# ===== 初始化会话状态 =====
if "papers" not in st.session_state:
    st.session_state.papers = []
if "students" not in st.session_state:
    st.session_state.students = {}
if "paper_counter" not in st.session_state:
    st.session_state.paper_counter = 0

# ===== 多格式解析器（核心升级） =====
def parse_uploaded_file(uploaded_file):
    """
    根据文件后缀自动调用对应解析器，返回文字块列表
    每个文字块格式： {"text": "文字内容", "location": {...} 或 None}
    """
    file_name = uploaded_file.name.lower()
    if file_name.endswith((".jpg", ".jpeg", ".png")):
        return baidu_ocr_with_location(uploaded_file.getvalue())
    elif file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file.getvalue())
    elif file_name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file.getvalue())
    else:
        st.error("不支持的文件格式，请上传 JPG/PNG/PDF/DOCX")
        return []

def baidu_ocr_with_location(image_bytes):
    """百度 OCR（返回文字及坐标）"""
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"
    # 获取 token
    token_host = "https://aip.baidubce.com/oauth/2.0/token"
    token_params = {
        "grant_type": "client_credentials",
        "client_id": BAIDU_API_KEY,
        "client_secret": BAIDU_SECRET_KEY
    }
    token_res = requests.get(token_host, params=token_params)
    access_token = token_res.json().get("access_token")

    img_base64 = base64.b64encode(image_bytes).decode()
    ocr_url = url + f"?access_token={access_token}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"image": img_base64}
    response = requests.post(ocr_url, headers=headers, data=data)
    result = response.json()
    words = []
    if "words_result" in result:
        for item in result["words_result"]:
            words.append({
                "text": item["words"],
                "location": item.get("location")
            })
    return words

def extract_text_from_pdf(file_bytes):
    """从 PDF 提取文本，每个非空行作为一个文字块（无坐标）"""
    lines = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.split("\n"):
                    stripped = line.strip()
                    if stripped:
                        lines.append({"text": stripped, "location": None})
    return lines

def extract_text_from_docx(file_bytes):
    """从 Word 提取文本，每个非空段落作为一个文字块"""
    doc = Document(io.BytesIO(file_bytes))
    blocks = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            blocks.append({"text": text, "location": None})
    return blocks

# ===== Gemini 调用（支持代理） =====
def call_gemini(prompt):
    proxies = None
    if GEMINI_PROXY:
        proxies = {"https": GEMINI_PROXY}
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(GEMINI_URL, headers=headers, json=data, proxies=proxies, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            st.error(f"Gemini 返回错误: {resp.status_code}")
            return None
    except Exception as e:
        st.error(f"无法连接 Gemini: {e}")
        return None

# ===== 切题逻辑 =====
def auto_split_questions(ocr_words):
    # 按纵坐标排序（无坐标的放前面）
    sorted_words = sorted(ocr_words, key=lambda w: w["location"]["top"] if w["location"] else 0)
    ptn = re.compile(r'^(\d{1,2})[\.、]')
    questions = []
    current_q = []
    current_num = None
    for w in sorted_words:
        text = w["text"].strip()
        m = ptn.match(text)
        if m:
            if current_q and current_num is not None:
                questions.append({"num": current_num, "blocks": current_q})
            current_num = m.group(1)
            current_q = [w]
        else:
            if current_q is not None:
                current_q.append(w)
            else:
                current_num = "1"
                current_q = [w]
    if current_q:
        questions.append({"num": current_num, "blocks": current_q})
    return questions

# ===== 批改标记绘制（仅图片可用） =====
def draw_grading_marks(image, questions):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    for q in questions:
        loc = q.get("location")
        if loc is None:
            continue
        x = loc["left"] + loc["width"] + 5
        y = loc["top"]
        if q.get("correct", True):
            draw.text((x, y), "√", fill="red", font=font)
        else:
            draw.ellipse([x-10, y-5, x+10, y+15], outline="red", width=2)
    return img

# ===== 保存试卷 =====
def save_paper(name, questions):
    paper = {
        "id": st.session_state.paper_counter,
        "name": name,
        "questions": questions
    }
    st.session_state.papers.append(paper)
    st.session_state.paper_counter += 1

# ===== UI =====
st.set_page_config(page_title="初中数学智能教学助手", layout="wide")
st.title("📐 初中数学智能教学助手 Pro")
st.markdown("支持图片、PDF、Word 上传，自动切题、错题追踪、智能反馈")

tabs = st.tabs(["📄 试卷管理与切题", "👩‍🎓 学生错题管理", "📊 全班课堂反馈", "💾 数据备份"])

# ---------- 试卷管理与切题 ----------
with tabs[0]:
    st.header("上传试卷，自动/手动切题")
    uploaded_file = st.file_uploader(
        "支持 PDF、Word、图片（点击下方选择文件）",
        type=["jpg", "jpeg", "png", "pdf", "docx"]
    )
    if uploaded_file:
        # 解析
        with st.spinner("正在解析文件内容..."):
            ocr_data = parse_uploaded_file(uploaded_file)
        if not ocr_data:
            st.error("未能提取到任何文字，请检查文件内容")
        else:
            file_type = uploaded_file.name.split(".")[-1].lower()
            # 如果是图片，展示原图
            if file_type in ["jpg", "jpeg", "png"]:
                image = Image.open(uploaded_file)
                st.image(image, caption="原始图片", use_column_width=True)
            else:
                st.info(f"已解析「{uploaded_file.name}」，共 {len(ocr_data)} 个文字块")
                image = None  # 无图像

            # 自动切题
            raw_questions = auto_split_questions(ocr_data)
            st.subheader("自动切题结果（可手动修改）")
            edited_questions = []
            for idx, q in enumerate(raw_questions):
                with st.expander(f"第 {q['num']} 题（{len(q['blocks'])}个块）"):
                    full_text = " ".join([b["text"] for b in q["blocks"]])
                    content = st.text_area(f"题 {q['num']} 完整内容", value=full_text, key=f"text_{idx}")
                    correct = st.radio("对/错", ["对", "错"], index=0, key=f"correct_{idx}") == "对"
                    # 定位第一个有效坐标
                    loc = None
                    for b in q["blocks"]:
                        if b.get("location"):
                            loc = b["location"]
                            break
                    edited_questions.append({
                        "num": q["num"],
                        "content": content,
                        "correct": correct,
                        "location": loc
                    })

            # 手动加题
            st.markdown("---")
            st.write("**手动补充题目**")
            m_num = st.text_input("题号", key="m_num")
            m_content = st.text_area("内容", key="m_content")
            if st.button("➕ 添加这道题"):
                edited_questions.append({
                    "num": m_num,
                    "content": m_content,
                    "correct": True,
                    "location": None
                })
                st.success("已添加")

            # 保存
            paper_name = st.text_input("试卷名称", value=f"试卷_{datetime.now().strftime('%m%d_%H%M')}")
            if st.button("💾 保存试卷到题库"):
                valid = [q for q in edited_questions if q["content"].strip()]
                if not valid:
                    st.warning("请至少保留一道有效题目")
                else:
                    save_paper(paper_name, valid)
                    st.success(f"「{paper_name}」已保存，题库现有 {len(st.session_state.papers)} 份试卷")
                    if image and st.checkbox("在图片上显示红色批改标记"):
                        marked = draw_grading_marks(image, valid)
                        st.image(marked, caption="批改预览", use_column_width=True)

# ---------- 学生错题管理 ----------
with tabs[1]:
    st.header("学生错题记录与导出")
    # ...（保持之前的逻辑，已支持从题库勾选错题，生成分析报告，导出 Word）...
    # 此处省略，因为您之前版本已有，无需改动。

# ---------- 全班课堂反馈 ----------
with tabs[2]:
    st.header("生成全班课堂反馈（AI 撰写）")
    lesson_file = st.file_uploader("📂 上传课堂材料（图片/PDF/Word）", type=["jpg","jpeg","png","pdf","docx"])
    lesson_text = st.text_area("✏️ 文字补充（可选）", height=120)
    if st.button("📢 生成反馈"):
        ocr_text = ""
        if lesson_file:
            with st.spinner("解析文件中..."):
                blocks = parse_uploaded_file(lesson_file)
                ocr_text = " ".join([b["text"] for b in blocks])
        full = lesson_text + "\n" + ocr_text
        if not full.strip():
            st.warning("请提供课堂内容")
        else:
            prompt = f"""你是一位初中数学老师，请根据以下内容生成班级课堂反馈（200-300字）：
一、授课内容概览（分点列出）
二、学生练习常见易错点分析
内容：{full}
"""
            with st.spinner("Gemini 写作中..."):
                feedback = call_gemini(prompt)
            if feedback:
                st.text_area("课堂反馈（可复制）", feedback, height=300)

# ---------- 数据备份 ----------
with tabs[3]:
    st.header("题库与数据导入/导出")
    # ...（同前，导出/导入 JSON）

# 注意：学生错题页和学生错题导出等逻辑与之前相同，这里因篇幅省略，实际使用时请补全。
