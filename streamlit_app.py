import streamlit as st
import requests
import base64
import json
import re
import io
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
from datetime import datetime
import tempfile
import os
from copy import deepcopy

# ===== 全局配置 =====
# 百度 OCR 密钥从 secrets 读取
BAIDU_API_KEY = st.secrets["BAIDU_API_KEY"]
BAIDU_SECRET_KEY = st.secrets["BAIDU_SECRET_KEY"]
# Gemini API 密钥
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
# 如果配置了代理，则使用（如 http://127.0.0.1:7890）
GEMINI_PROXY = st.secrets.get("GEMINI_PROXY", None)

# Gemini API 基础 URL（直接调用）
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

# ===== 初始化 session_state（题库、学生记录） =====
if "papers" not in st.session_state:
    st.session_state.papers = []  # 每份试卷：{id, name, questions:[{num, content, correct}]}
if "students" not in st.session_state:
    st.session_state.students = {}  # 学生姓名 -> {wrong:[{paper_id, num, content}]}
if "paper_counter" not in st.session_state:
    st.session_state.paper_counter = 0

# ===== 百度 OCR（含位置） =====
def baidu_ocr_with_location(image_bytes):
    """调用百度通用文字识别（含位置），返回每个文字块的坐标和文字"""
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"
    # 获取 access_token
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

    words_info = []
    if "words_result" in result:
        for item in result["words_result"]:
            words_info.append({
                "text": item["words"],
                "location": item["location"]  # {left, top, width, height}
            })
    return words_info

# ===== 调用 Gemini（支持代理） =====
def call_gemini(prompt):
    """直接 HTTP 请求，可通过 GEMINI_PROXY 设置代理"""
    proxies = None
    if GEMINI_PROXY:
        proxies = {"https": GEMINI_PROXY}
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        resp = requests.post(GEMINI_URL, headers=headers, json=data, proxies=proxies, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            st.error(f"Gemini 返回错误: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        st.error(f"无法连接 Gemini: {e}")
        return None

# ===== 自动切题（基于题号） =====
def auto_split_questions(ocr_words):
    """
    根据题号（如"1."、"2."）将文字块分组，返回题目列表
    返回：[[text_block1, text_block2], ...]
    """
    # 把文字块按纵坐标排序（从上到下，从左到右）
    sorted_words = sorted(ocr_words, key=lambda w: (w["location"]["top"], w["location"]["left"]))
    
    # 检测题号的正则
    ptn = re.compile(r'^(\d{1,2})[\.、]')
    questions = []
    current_q = []
    current_num = None
    
    for w in sorted_words:
        text = w["text"].strip()
        m = ptn.match(text)
        if m:
            # 开始新题目
            if current_q and current_num is not None:
                questions.append({"num": current_num, "blocks": current_q})
            current_num = m.group(1)
            current_q = [w]
        else:
            if current_q is not None:
                current_q.append(w)
            else:
                # 开头没有题号的情况，作为第一题
                current_num = "1"
                current_q = [w]
    # 最后一题
    if current_q:
        questions.append({"num": current_num, "blocks": current_q})
    return questions

# ===== 绘制红色批改标记（在图像上画 √/○） =====
def draw_grading_marks(image, questions):
    """
    传入 PIL Image 和题目列表（含 correct 字段），在原图上绘制对错符号
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    # 尝试加载字体，如果没有就默认
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    for q in questions:
        if "location" in q:  # 需要有坐标信息
            loc = q["location"]
            x = loc["left"] + loc["width"] + 5
            y = loc["top"]
            if q.get("correct", True):
                # 绿色 √
                draw.text((x, y), "√", fill="red", font=font)
            else:
                # 红色 ○
                draw.ellipse([x-10, y-5, x+10, y+15], outline="red", width=2)
    return img

# ===== 保存试卷到题库 =====
def save_paper(name, questions):
    paper = {
        "id": st.session_state.paper_counter,
        "name": name,
        "questions": questions  # [{num, content, correct}]
    }
    st.session_state.papers.append(paper)
    st.session_state.paper_counter += 1

# ===== 页面布局 =====
st.set_page_config(page_title="初中数学教学助手 Pro", layout="wide")
st.title("📐 初中数学教学助手 Pro")
st.markdown("—— 智能切题·错题追踪·个性化反馈 ——")

tabs = st.tabs([
    "📄 试卷管理与切题",
    "👩‍🎓 学生错题管理",
    "📊 全班课堂反馈",
    "💾 数据导入/导出"
])

# ================= 试卷管理 =================
with tabs[0]:
    st.header("1. 上传试卷并切题")
    uploaded_file = st.file_uploader("选择试卷图片（jpg/png）", type=["jpg","jpeg","png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        img_bytes = uploaded_file.getvalue()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("原始图片")
            st.image(image, use_column_width=True)

        # OCR 识别（含位置）
        with st.spinner("正在识别文字与位置..."):
            ocr_data = baidu_ocr_with_location(img_bytes)
        
        # 自动切题
        raw_questions = auto_split_questions(ocr_data)
        st.session_state.raw_questions = raw_questions
        st.session_state.ocr_data = ocr_data

        # 显示自动切题结果
        st.subheader("自动切题结果（可手动调整）")
        edited_questions = []
        for idx, q in enumerate(raw_questions):
            with st.expander(f"第 {q['num']} 题（{len(q['blocks'])}个文字块）"):
                full_text = " ".join([b["text"] for b in q["blocks"]])
                content = st.text_area(f"题目内容 （题号：{q['num']}）", full_text, key=f"q_text_{idx}")
                correct = st.radio("对错", ["对", "错"], index=0, key=f"correct_{idx}") == "对"
                edited_questions.append({
                    "num": q["num"],
                    "content": content,
                    "correct": correct,
                    "location": q["blocks"][0]["location"]  # 取第一个文字块位置用于绘图
                })

        # 手动添加/删除题目
        st.markdown("---")
        st.write("**手动补充题目**")
        manual_num = st.text_input("题号", "")
        manual_content = st.text_area("内容", "")
        if st.button("添加此题目"):
            edited_questions.append({
                "num": manual_num,
                "content": manual_content,
                "correct": True,
                "location": None
            })
            st.success("已添加")

        # 保存试卷
        paper_name = st.text_input("试卷名称（例如：2025年3月月考）", value=f"试卷_{datetime.now().strftime('%m%d_%H%M')}")
        if st.button("💾 保存试卷到题库"):
            # 过滤掉空题
            valid_questions = [q for q in edited_questions if q["content"].strip()]
            if not valid_questions:
                st.warning("请至少保留一道有效题目")
            else:
                save_paper(paper_name, valid_questions)
                st.success(f"试卷「{paper_name}」已保存！题库中有 {len(st.session_state.papers)} 份试卷。")
                # 可选：在图片上绘制批改标记
                if st.checkbox("在图片上显示红色批改标记"):
                    marked_img = draw_grading_marks(image, valid_questions)
                    st.image(marked_img, caption="批改效果预览", use_column_width=True)

# ================= 学生错题管理 =================
with tabs[1]:
    st.header("2. 学生错题记录")
    # 学生姓名管理
    all_student_names = list(st.session_state.students.keys())
    student_name = st.text_input("输入学生姓名", "")
    if student_name and student_name not in st.session_state.students:
        if st.button(f"新建学生「{student_name}」"):
            st.session_state.students[student_name] = {"wrong": []}
            st.success(f"已创建学生 {student_name} 的档案")
    
    selected_student = st.selectbox("选择已有学生", options=[""] + all_student_names, index=0)
    student_name = student_name or selected_student  # 优先使用输入框

    if student_name and student_name in st.session_state.students:
        st.subheader(f"📋 {student_name} 的错题集")
        # 显示已有错题
        wrong_list = st.session_state.students[student_name]["wrong"]
        if wrong_list:
            df = pd.DataFrame(wrong_list)
            st.dataframe(df)
        else:
            st.info("暂无错题记录")

        # 从题库添加错题
        st.markdown("---")
        st.write("### 从已保存试卷中录入错题")
        if st.session_state.papers:
            paper_names = [p["name"] for p in st.session_state.papers]
            chosen_paper_name = st.selectbox("选择试卷", paper_names)
            chosen_paper = next(p for p in st.session_state.papers if p["name"] == chosen_paper_name)
            
            # 显示该试卷题目，让用户勾选错题
            st.write(f"试卷「{chosen_paper_name}」共有 {len(chosen_paper['questions'])} 题，请勾选错题：")
            selected_indices = []
            for i, q in enumerate(chosen_paper["questions"]):
                col1, col2 = st.columns([0.05, 0.95])
                with col1:
                    checked = st.checkbox("", key=f"sel_{chosen_paper['id']}_{i}")
                with col2:
                    st.write(f"**{q['num']}.** {q['content'][:100]}...")
                if checked:
                    selected_indices.append(i)
            
            if st.button("📥 保存至该生错题库"):
                for idx in selected_indices:
                    q = chosen_paper["questions"][idx]
                    # 避免重复添加（简单判断相同的paper_id和num）
                    existing = [e for e in wrong_list if e["paper_id"]==chosen_paper["id"] and e["num"]==q["num"]]
                    if not existing:
                        wrong_list.append({
                            "paper_id": chosen_paper["id"],
                            "paper_name": chosen_paper_name,
                            "num": q["num"],
                            "content": q["content"]
                        })
                st.session_state.students[student_name]["wrong"] = wrong_list
                st.success(f"已添加 {len(selected_indices)} 道错题")
        else:
            st.warning("题库为空，请先上传并保存试卷")

        # 生成个性化分析（调用Gemini）
        if st.button("🔍 生成该生个性化错题分析报告"):
            if wrong_list:
                all_wrong_text = "\n".join([f"{w['num']}. {w['content']}" for w in wrong_list])
                prompt = f"""你是一位初中数学老师。请根据以下学生的所有错题，写一份200字左右的个性化分析报告，包括：
- 主要薄弱知识点
- 建议的复习方向
- 几句鼓励的话
学生姓名：{student_name}
错题列表：
{all_wrong_text}
"""
                with st.spinner("Gemini 正在分析..."):
                    report = call_gemini(prompt)
                if report:
                    st.text_area("分析报告", report, height=200)
            else:
                st.warning("请先录入错题")

        # 下载错题集
        if wrong_list and st.button("⬇️ 导出 Word 错题集"):
            from docx import Document
            doc = Document()
            doc.add_heading(f"{student_name} 数学错题集", 0)
            for w in wrong_list:
                doc.add_paragraph(f"试卷：{w['paper_name']}  题号：{w['num']}")
                doc.add_paragraph(w['content'])
                doc.add_paragraph("")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
            doc.save(tmp.name)
            with open(tmp.name, "rb") as f:
                st.download_button("下载错题集.docx", f, file_name=f"{student_name}_错题集.docx")
            os.unlink(tmp.name)

# ================= 全班课堂反馈 =================
with tabs[2]:
    st.header("3. 生成全班课堂反馈")
    st.write("上传本节课的板书/课件照片，或输入文字描述，Gemini 将生成可直接发班群的反馈（格式参考示例）")

    lesson_img = st.file_uploader("📷 上传课堂照片/文件", type=["jpg","jpeg","png","pdf"], key="lesson_img")
    lesson_text = st.text_area("✏️ 文字补充（可选）", height=100)

    if st.button("📢 生成班级反馈"):
        ocr_text = ""
        if lesson_img:
            with st.spinner("识别图片文字..."):
                ocr_text = baidu_ocr_with_location(lesson_img.getvalue())
                ocr_text = " ".join([w["text"] for w in ocr_text])
        full_desc = lesson_text + "\n" + ocr_text

        prompt = f"""你是一位初中数学老师。请根据以下课堂内容描述，生成一份班级课堂反馈，格式如下：
一、 授课内容概览（概括本节课知识体系，分点列出）
二、 学生练习常见易错点分析（列出3-4个典型错误，用通俗语言说明）
要求200-300字，语气专业亲切。

课堂内容素材：
{full_desc}
"""
        with st.spinner("Gemini 写作中..."):
            feedback = call_gemini(prompt)
        if feedback:
            st.text_area("课堂反馈（可复制到班群）", feedback, height=300)

# ================= 数据导入/导出 =================
with tabs[3]:
    st.header("4. 题库与学生数据导入/导出")
    st.markdown("为了保证数据不会丢失，请定期导出数据库为 JSON 文件，下次使用前导入即可。")

    # 导出
    if st.button("⬇️ 导出全部数据"):
        export_data = {
            "papers": st.session_state.papers,
            "students": st.session_state.students,
            "paper_counter": st.session_state.paper_counter
        }
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
        st.download_button(
            "下载数据备份.json",
            json_str,
            file_name="教学助手数据.json",
            mime="application/json"
        )

    # 导入
    st.write("导入之前导出的 JSON 文件（会覆盖当前数据）")
    uploaded_json = st.file_uploader("上传数据文件", type="json", key="json_upload")
    if uploaded_json:
        try:
            imported = json.load(uploaded_json)
            st.session_state.papers = imported.get("papers", [])
            st.session_state.students = imported.get("students", {})
            st.session_state.paper_counter = imported.get("paper_counter", 0)
            st.success("数据导入成功！请刷新页面或切换到其他选项卡查看。")
        except:
            st.error("文件格式错误")
