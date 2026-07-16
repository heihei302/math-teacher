import streamlit as st
import requests
import base64
import json
import cv2
import numpy as np
import re
from PIL import Image
from io import BytesIO
from datetime import datetime
import sqlite3
import pandas as pd

# ──────────────────────────────────────
# 百度OCR配置：请把下面的 xxxx 替换成你的真实密钥
# ──────────────────────────────────────
BAIDU_API_KEY = "bZTSq24Cgos2yPqJl5dGHaMg"          # 改这里
BAIDU_SECRET_KEY = "eUOIIFCEV76xljrk8EKU9pS5F5jOueYD"    # 改这里

def get_baidu_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": BAIDU_API_KEY,
        "client_secret": BAIDU_SECRET_KEY
    }
    return requests.post(url, params=params).json().get("access_token")

def baidu_ocr_general(image_bytes):
    """通用文字识别（纯文本题用）"""
    token = get_baidu_token()
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token={token}"
    img_base64 = base64.b64encode(image_bytes).decode()
    r = requests.post(url, data={"image": img_base64},
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
    return r.json()

def baidu_ocr_formula(image_bytes):
    """数学公式识别（使用‘网络图片文字识别’里的公式模式）"""
    token = get_baidu_token()
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/formula?access_token={token}"
    img_base64 = base64.b64encode(image_bytes).decode()
    r = requests.post(url, data={"image": img_base64},
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
    return r.json()

# ──────────────────────────────────────
# 数据库初始化（SQLite，自动创建）
# ──────────────────────────────────────
conn = sqlite3.connect("mistakes.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS mistakes
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              student TEXT,
              question TEXT,
              knowledge TEXT,
              date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

# ──────────────────────────────────────
# 知识点分类规则（可自行扩充）
# ──────────────────────────────────────
def classify_knowledge(text):
    text = text.lower()
    if any(w in text for w in ['方程', '解方程', '未知数', '等式', '2x', '3x']):
        return '一元一次方程'
    if any(w in text for w in ['不等式', '>', '<', '≥', '≤']):
        return '不等式'
    if any(w in text for w in ['三角形', '全等', '勾股', '角度', '边长', '正弦', '余弦']):
        return '三角形与勾股定理'
    if any(w in text for w in ['函数', '一次函数', '正比例', 'y=', '图象', '抛物线']):
        return '函数'
    if any(w in text for w in ['圆', '弧', '弦', '圆心角', '切线']):
        return '圆的性质'
    if any(w in text for w in ['概率', '统计', '平均数', '中位数', '方差']):
        return '统计与概率'
    if any(w in text for w in ['整式', '因式分解', '平方差', '完全平方']):
        return '整式与因式分解'
    if any(w in text for w in ['分式']):
        return '分式'
    return '其他知识点'

# ──────────────────────────────────────
# 题目电子化核心函数：用百度OCR识别文字和公式
# ──────────────────────────────────────
def digitize_question(img_pil):
    """将一道题的图片转成电子文本（文字+LaTeX公式）"""
    img_bytes = BytesIO()
    img_pil.save(img_bytes, format='PNG')
    img_data = img_bytes.getvalue()

    # 先用通用OCR获取文字
    res = baidu_ocr_general(img_data)
    words = res.get("words_result", [])
    text_lines = [w['words'] for w in words]
    # 再用公式OCR检测可能存在的公式
    formula_res = baidu_ocr_formula(img_data)
    formulas = formula_res.get("words_result", [])
    formula_lines = [f['words'] for f in formulas]  # 返回的是 LaTeX 格式

    # 简单合并：把公式按位置插入（这里为了演示，只把公式附加到文本后）
    combined = ""
    if text_lines:
        combined += "题目：" + " ".join(text_lines)
    if formula_lines:
        combined += "\n公式部分：" + " ; ".join(formula_lines)
    return combined.strip()

# ──────────────────────────────────────
# 红色笔迹错题检测（OpenCV）
# ──────────────────────────────────────
def detect_red_marks(image_cv):
    hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 100]

def is_question_wrong(question_pil, red_rects):
    """把题目图片与全图红色区域比对"""
    # 这里简化：如果题目区域与红色矩形有重叠就算错
    # 实际需要坐标匹配，但Streamlit处理上传图片时没有统一坐标，所以我们用另一个方法：
    # 直接在整个图片上判断，每个题目我们会先切割出来，再调用此函数。
    # 更稳健的方案：直接对整张卷子做切割，然后逐一判断。
    # 为了演示清晰，我们只针对整张卷子使用，返回整个卷子是否有红色标记。
    return len(red_rects) > 0

# 题目切割（基于题号识别，使用简单规则）
def split_questions(img_pil):
    """将试卷分割为单个题目图片"""
    img_cv = np.array(img_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # 找水平投影，分割行
    # 这里简化：用百度OCR的段落检测结果进行切割
    # 因为自己写切割鲁棒性不够，我们改用百度OCR的段落信息（需要升级API）
    # 返回一个列表，每个元素是一道题的图片区域
    # 注：百度高精度OCR可以返回段落坐标，但免费版只返回文字，无坐标。
    # 作为演示，我们提供一个手动方法：用户手动框选错题区域，或提供更完整的坐标识别。
    # 为了能让程序跑通，我们采用整张试卷电子化，不分割。
    return [img_pil]  # 返回整张图

# ──────────────────────────────────────
# 家长反馈生成模板（成熟稳重教师口吻）
# ──────────────────────────────────────
def parent_feedback(student_name, mistake_list):
    if not mistake_list:
        return f"{student_name}家长，孩子本次练习完成得很好，准确率高，值得表扬！"
    kp = list(set(m['knowledge'] for m in mistake_list))
    lines = [f"{student_name}家长您好！本次数学练习孩子共错了{len(mistake_list)}道题，涉及知识点：{'、'.join(kp)}。"]
    for i, m in enumerate(mistake_list, 1):
        lines.append(f"错题{i}：{m['question'][:50]}... 【{m['knowledge']}】需要加强。")
    lines.append("\n建议家长协助：")
    if '一元一次方程' in kp:
        lines.append("✅ 每天做2道解方程，强调步骤规范。")
    if '三角形与勾股定理' in kp:
        lines.append("✅ 让孩子动手画图，结合图形理解定理。")
    if '函数' in kp:
        lines.append("✅ 多描点画函数图像，感受变量关系。")
    lines.append("\n如有需要，我可以推荐针对性练习。我们一起帮助孩子稳步提升！")
    return "\n".join(lines)

# ─── 班级反馈模板 ───
def class_feedback(lesson_text, summary):
    return f"""📚【数学课堂反馈】
本节课内容：{lesson_text}
全班练习概况：{summary}

🌟 亮点：大部分同学积极参与，解题思路清晰。
⚠️ 存在问题：部分同学计算粗心，忘记检验答案。
📌 课后建议：整理错题，完成对应章节练习，下节课课前检查。
让我们一起关注孩子的成长！"""

# ──────────────────────────────────────
# Streamlit 前端界面
# ──────────────────────────────────────
st.set_page_config(page_title="数学教学助手", layout="wide")
st.title("👩‍🏫 初中数学教学智能助手")

# 侧边栏：学生错题集查询
with st.sidebar:
    st.subheader("📖 错题集查询")
    student_query = st.text_input("输入学生姓名查询错题")
    if st.button("查询"):
        if student_query:
            df = pd.read_sql_query(
                "SELECT question, knowledge, date FROM mistakes WHERE student=? ORDER BY date DESC",
                conn, params=(student_query,)
            )
            if df.empty:
                st.info("该学生暂无错题记录。")
            else:
                st.dataframe(df)
                st.download_button("下载错题CSV", df.to_csv(index=False), file_name=f"{student_query}_错题集.csv")

# 主功能区
tab1, tab2, tab3 = st.tabs(["📝 学生试卷分析", "🏫 班级课堂反馈", "📚 题目电子化入库"])

# ─────────────── 功能1：学生试卷分析 ───────────────
with tab1:
    st.subheader("上传学生试卷，自动识别错题并生成家长反馈")
    student_name = st.text_input("学生姓名", key="sname")
    uploaded_files = st.file_uploader("选择试卷图片（可多选）", type=["png","jpg","jpeg"], accept_multiple_files=True)
    if uploaded_files and student_name:
        all_mistakes = []
        for file in uploaded_files:
            with st.spinner(f"正在分析 {file.name} …"):
                # 读取图片
                img_bytes = file.getvalue()
                img_pil = Image.open(BytesIO(img_bytes))
                # 检测红色标记（错题）
                img_cv = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                red_rects = detect_red_marks(img_cv)
                has_red = len(red_rects) > 0
                # 电子化整张卷子（包括公式）
                full_text = digitize_question(img_pil)
                # 知识点分类
                kp = classify_knowledge(full_text)
                # 如果存在红色标记，视为错题（整张卷子有错就全收，简单处理）
                if has_red:
                    all_mistakes.append({
                        "question": full_text,
                        "knowledge": kp,
                        "img": img_pil
                    })
                st.image(img_pil, caption=f"{file.name} 红色标记{len(red_rects)}处", width=300)
        # 保存错题到数据库
        for m in all_mistakes:
            c.execute("INSERT INTO mistakes (student, question, knowledge) VALUES (?,?,?)",
                      (student_name, m['question'], m['knowledge']))
            conn.commit()
        st.success(f"分析完成！共收录 {len(all_mistakes)} 道错题。")
        # 生成反馈
        if all_mistakes:
            feedback = parent_feedback(student_name, all_mistakes)
            st.text_area("📲 家长反馈（可直接复制）", feedback, height=200)

# ─────────────── 功能2：班级课堂反馈 ───────────────
with tab2:
    st.subheader("生成本节课的班级群反馈")
    lesson_text = st.text_area("授课内容摘要（例如：一元一次方程的应用）")
    lesson_images = st.file_uploader("上传课堂板书/课件截图（可选）", type=["png","jpg"], accept_multiple_files=True)
    if lesson_images:
        for img in lesson_images:
            st.image(img, width=300)
    mistake_summary = st.text_area("全班练习情况汇总（例如：解方程错8人，应用题审题不清12人）")
    if st.button("生成班级反馈") and lesson_text:
        if mistake_summary:
            summary = mistake_summary
        else:
            summary = "本次练习整体情况良好，少数同学需要加强计算准确性。"
        feedback = class_feedback(lesson_text, summary)
        st.text_area("📢 班级群文案", feedback, height=250)

# ─────────────── 功能3：题目电子化入库 ───────────────
with tab3:
    st.subheader("将任意题目照片转为电子文本并自动分类")
    q_file = st.file_uploader("上传题目图片（单题）", type=["png","jpg"])
    if q_file:
        img = Image.open(q_file)
        st.image(img, width=400)
        if st.button("电子化并分类"):
            with st.spinner("识别中..."):
                text = digitize_question(img)
                kp = classify_knowledge(text)
            st.subheader("识别结果")
            st.text_area("题目文本", text, height=150)
            st.info(f"知识点分类：{kp}")
            if st.button("保存到题库"):
                c.execute("INSERT INTO mistakes (student, question, knowledge) VALUES (?,?,?)",
                          ("公共题库", text, kp))
                conn.commit()
                st.success("已保存！")

st.caption("💡 提示：所有错题会自动记录，可在左侧查询。")
