import streamlit as st
import requests
import base64
import json
import cv2
import numpy as np
import re
from PIL import Image, ImageDraw
from io import BytesIO
from datetime import datetime
import sqlite3
import pandas as pd

# ──────────────────────────────────────
# ‼️ 百度OCR密钥：请替换成你自己的真实值
# ──────────────────────────────────────
BAIDU_API_KEY = "bZTSq24Cgos2yPqJl5dGHaMg"
BAIDU_SECRET_KEY = "eUOIIFCEV76xljrk8EKU9pS5F5jOueYD"

def get_baidu_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": BAIDU_API_KEY,
        "client_secret": BAIDU_SECRET_KEY
    }
    return requests.post(url, params=params).json().get("access_token")

def baidu_ocr_general(image_bytes, location=False):
    """通用文字识别（可返回坐标）"""
    token = get_baidu_token()
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic?access_token={token}"
    params = {"image": base64.b64encode(image_bytes).decode()}
    if location:
        url += "&location=true"   # 请求坐标
        url += "&detect_direction=true"  # 自动检测方向
    r = requests.post(url, data=params,
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
    return r.json()

def baidu_ocr_formula(image_bytes):
    """数学公式识别"""
    token = get_baidu_token()
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/formula?access_token={token}"
    params = {"image": base64.b64encode(image_bytes).decode()}
    r = requests.post(url, data=params,
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
    return r.json()

# ──────────────────────────────────────
# 数据库初始化
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
# 知识点分类规则（你可以自由增删）
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
# 🔧 核心升级：精准分割题目
# ──────────────────────────────────────
def split_questions(img_pil):
    """
    把试卷图片按题号切成单道题目的图像列表
    返回：[(题目图片, 题目文本), ...]
    """
    img_cv = np.array(img_pil)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    img_bytes = BytesIO()
    img_pil.save(img_bytes, format='PNG')
    img_data = img_bytes.getvalue()

    # 1. 获取所有文字块及坐标
    res = baidu_ocr_general(img_data, location=True)
    words_info = res.get("words_result", [])
    if not words_info:
        return [(img_pil, "")]   # 如果识别不到，返回整张图

    # 2. 按行整理（将同一水平线上的词合并成一行）
    lines = []   # 每条记录：平均y, 所有词的信息列表
    for w in words_info:
        text = w.get("words", "")
        loc = w.get("location", {})
        if not loc:
            continue
        # 计算四个顶点的平均y坐标
        ys = [loc['top'], loc['top']+loc['height'], loc['top']+loc['height'], loc['top']]
        # location 可能格式是 left, top, width, height
        if 'left' in loc and 'top' in loc and 'width' in loc and 'height' in loc:
            left = loc['left']
            top = loc['top']
            width = loc['width']
            height = loc['height']
            avg_y = top + height/2
        else:
            # 旧格式：四个顶点
            pts = loc  # 类似 {'x':...,'y':...} 四个点，但 general_basic 返回的是 left,top,width,height
            # 为了兼容，以防万一使用中点
            avg_y = sum([p.get('y', 0) for p in pts])/4 if isinstance(pts, list) else 0
        lines.append((avg_y, left, top, width, height, text))

    # 按照y坐标排序（从上到下）
    lines.sort(key=lambda x: x[0])

    # 合并同一行（y坐标差小于15个像素的视为一行）
    merged_rows = []
    current_row = []
    current_y = None
    threshold = 15
    for item in lines:
        y = item[0]
        if current_y is None or abs(y - current_y) < threshold:
            current_row.append(item)
            current_y = y
        else:
            merged_rows.append((current_y, current_row))
            current_row = [item]
            current_y = y
    if current_row:
        merged_rows.append((current_y, current_row))

    # 3. 找题号行
    question_starts = []
    for idx, (row_y, row_items) in enumerate(merged_rows):
        # 将该行所有文本拼在一起
        row_text = " ".join([it[5] for it in row_items])
        # 检测题号：数字后跟 . 或 、（比如 "1." "2." "5、"）
        if re.match(r'^\s*(\d+)[\.\、]', row_text.strip()):
            question_starts.append(idx)

    if not question_starts:
        return [(img_pil, "")]   # 找不到题号，返回整张图

    # 4. 根据题号行切出题目区域
    questions = []
    h, w = img_pil.height, img_pil.width
    for i, start_idx in enumerate(question_starts):
        # 确定本题区域：从本题号行y_min到下一题号行y_min（或图像底边）
        start_y = int(merged_rows[start_idx][0] - 10)  # 向上留点边距
        if i + 1 < len(question_starts):
            next_idx = question_starts[i+1]
            next_y = int(merged_rows[next_idx][0] - 10)
            end_y = next_y
        else:
            end_y = h

        # 考虑左右边界（稍微扩展）
        x1 = 0
        x2 = w
        # 裁剪并保存为图片
        crop_img = img_pil.crop((x1, max(0, start_y), x2, min(h, end_y)))
        # 对该裁剪区域进行OCR，提取文本（使用通用OCR即可）
        crop_bytes = BytesIO()
        crop_img.save(crop_bytes, format='PNG')
        crop_data = crop_bytes.getvalue()
        ocr_res = baidu_ocr_general(crop_data, location=False)
        crop_text = " ".join([w['words'] for w in ocr_res.get("words_result", [])])
        questions.append((crop_img, crop_text.strip()))

    return questions

# ──────────────────────────────────────
# 题目电子化（文字+公式）
# ──────────────────────────────────────
def digitize_question(img_pil):
    img_bytes = BytesIO()
    img_pil.save(img_bytes, format='PNG')
    img_data = img_bytes.getvalue()

    # 文字OCR
    res = baidu_ocr_general(img_data, location=False)
    words = res.get("words_result", [])
    text_lines = [w['words'] for w in words]
    # 公式OCR
    formula_res = baidu_ocr_formula(img_data)
    formulas = formula_res.get("words_result", [])
    formula_lines = [f['words'] for f in formulas]

    combined = ""
    if text_lines:
        combined += "题目：" + " ".join(text_lines)
    if formula_lines:
        combined += "\n公式部分：" + " ; ".join(formula_lines)
    return combined.strip()

# ──────────────────────────────────────
# 红色笔迹检测
# ──────────────────────────────────────
def detect_red_marks(image_cv):
    """返回红色区域的矩形列表"""
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
    rects = []
    for cont in contours:
        if cv2.contourArea(cont) > 100:   # 过滤小噪点
            rects.append(cv2.boundingRect(cont))
    return rects

# ──────────────────────────────────────
# 家长反馈模板
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

def class_feedback(lesson_text, summary):
    return f"""📚【数学课堂反馈】
本节课内容：{lesson_text}
全班练习概况：{summary}

🌟 亮点：大部分同学积极参与，解题思路清晰。
⚠️ 存在问题：部分同学计算粗心，忘记检验答案。
📌 课后建议：整理错题，完成对应章节练习，下节课课前检查。
让我们一起关注孩子的成长！"""

# ──────────────────────────────────────
# 🖥 Streamlit 界面
# ──────────────────────────────────────
st.set_page_config(page_title="数学教学助手 Pro", layout="wide")
st.title("👩‍🏫 初中数学教学智能助手（精准分割版）")

# 侧边栏：错题集
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

tab1, tab2, tab3 = st.tabs(["📝 学生试卷分析", "🏫 班级课堂反馈", "📚 题目电子化入库"])

# ───── 功能1：学生试卷分析（升级精准切割）─────
with tab1:
    st.subheader("上传学生试卷，逐题分析对错并生成家长反馈")
    student_name = st.text_input("学生姓名", key="sname")
    uploaded_files = st.file_uploader("选择试卷图片（可多选）", type=["png","jpg","jpeg"], accept_multiple_files=True)
    if uploaded_files and student_name:
        all_mistakes = []
        for file in uploaded_files:
            with st.spinner(f"正在分析 {file.name} …"):
                img_bytes = file.getvalue()
                img_pil = Image.open(BytesIO(img_bytes))
                # 分割成题目列表
                question_pieces = split_questions(img_pil)
                st.write(f"📑 {file.name} 共检测到 {len(question_pieces)} 道题目")
                # 对每道题单独处理
                for q_idx, (q_img, q_text) in enumerate(question_pieces, start=1):
                    # 转为cv2格式检测红色标记
                    q_cv = cv2.cvtColor(np.array(q_img), cv2.COLOR_RGB2BGR)
                    red_rects = detect_red_marks(q_cv)
                    if len(red_rects) > 0:
                        # 电子化完整题目（文字+公式）
                        full_text = digitize_question(q_img)
                        kp = classify_knowledge(full_text)
                        all_mistakes.append({
                            "question": full_text,
                            "knowledge": kp,
                            "img": q_img
                        })
                        st.image(q_img, caption=f"第{q_idx}题 ❌ 错题", width=250)
                    else:
                        st.image(q_img, caption=f"第{q_idx}题 ✓ 正确", width=250)
        # 保存错题到数据库
        for m in all_mistakes:
            c.execute("INSERT INTO mistakes (student, question, knowledge) VALUES (?,?,?)",
                      (student_name, m['question'], m['knowledge']))
            conn.commit()
        st.success(f"分析完成！收录 {len(all_mistakes)} 道错题。")
        if all_mistakes:
            feedback = parent_feedback(student_name, all_mistakes)
            st.text_area("📲 家长反馈（可直接复制）", feedback, height=200)

# ───── 功能2：班级课堂反馈 ─────
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

# ───── 功能3：题目电子化入库 ─────
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

st.caption("💡 提示：题库自动记录错题，左侧查询。试卷分析现已支持逐题精准切割。")
