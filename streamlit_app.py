import streamlit as st
import google.generativeai as genai
import requests
import base64
import json
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
import re
from docx import Document
from docx.shared import Pt, Inches
import tempfile
import os
from typing import List, Dict

# ===== 配置：从 Streamlit Cloud 的 Secrets 中读取密钥 =====
BAIDU_API_KEY = st.secrets["BAIDU_API_KEY"]
BAIDU_SECRET_KEY = st.secrets["BAIDU_SECRET_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# 配置 Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')  # 文本模型

# ===== 百度 OCR 函数 =====
def baidu_ocr(image_bytes):
    """调用百度通用文字识别（高精度版）"""
    url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
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
            words.append(item["words"])
    return "\n".join(words)

# ===== Gemini 通用调用函数 =====
def gemini_generate(prompt, fallback=""):
    """调用 Gemini 生成文本，失败时返回备用内容"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"Gemini 调用失败：{e}，使用备用内容。")
        return fallback

# ===== 错题分析与知识点提取 =====
def analyze_test_paper(ocr_text, class_level="八年级"):
    """
    让 Gemini 从 OCR 文字中提取题目、判断对错、分析知识点
    返回: {
        "questions": [ { "number": 1, "content": "...", "correct": true, "knowledge": "代数" } ],
        "summary": "整体情况描述"
    }
    """
    prompt = f"""你是一位经验丰富的{ class_level }数学老师。下面是从一张已经批改过的学生试卷上识别出的文字（可能包含老师打的对错标记 √ 或 × 以及得分）。
请严格按照以下要求处理：

1. 提取试卷中的所有题目，并为每道题给出：
   - 题号（如果识别不出用 "未知"）
   - 题目完整内容（包括学生的作答痕迹）
   - 对错判断（根据识别到的 √ 或 ×，或根据得分标记推理，无法判断时标注为"未知"）
   - 所属知识点（例如：一元一次方程、平面几何、因式分解等，用简洁的词组表示）

2. 对整体情况给出100字左右的总结，包括：
   - 完成较好的知识点
   - 需要加强的知识点

请以 JSON 数组格式返回，格式如下（不要包含其他文字）：
{{
  "questions": [
    {{"number": "1", "content": "题目内容...", "correct": true, "knowledge": "代数"}},
    ...
  ],
  "summary": "整体完成情况总结..."
}}

OCR 文字：
{ocr_text}
"""
    result_text = gemini_generate(prompt, fallback='{"questions":[], "summary":"无法分析"}')
    # 尝试解析 JSON
    try:
        # 去掉可能的 markdown 代码块标记
        cleaned = re.sub(r'```json|```', '', result_text).strip()
        data = json.loads(cleaned)
        return data
    except:
        st.error("Gemini 返回格式异常，请重试。")
        return {"questions": [], "summary": "解析失败"}

# ===== 家长反馈生成 =====
def parent_feedback(wrong_questions, knowledge_stats, class_level):
    """根据错题列表和知识点统计生成给家长的话"""
    prompt = f"""你是一位成熟稳重的初中数学老师。请根据以下学生的错题情况，写一段可以直接发给家长的反馈（约120-180字）：
要求：
- 语气鼓励、亲切，先肯定优点，再指出需要加强的地方，并提供家庭辅导建议。
- 使用称呼“家长您好”。
- 提及具体的薄弱知识点。

学生年级：{ class_level }
错题涉及的知识点：{', '.join(knowledge_stats)}
错题数量：{len(wrong_questions)}
典型错题示例：{wrong_questions[0]['content'] if wrong_questions else '无'}
"""
    return gemini_generate(prompt, fallback="家长您好，孩子本次作业完成认真，部分题目需要加强练习，请查看错题集。")

# ===== 班级课堂反馈生成 =====
def class_feedback(lesson_content_text, test_summary, class_level):
    """结合本节课内容和试卷整体情况生成班群反馈"""
    prompt = f"""你是一位{ class_level }数学老师。请根据以下信息，生成一段可以发在班级群里的“课堂反馈”，内容需包含：
1. 本节课授课内容概述（根据提供的图片/文字描述）
2. 本节课的重难点
3. 结合学生上次测试的整体情况，指出需要共同关注的问题

要求语言简洁正式，方便家长了解，总字数200-300字。

=== 本节课内容描述 ===
{ lesson_content_text }

=== 上次测试整体情况 ===
{ test_summary }
"""
    return gemini_generate(prompt, fallback="今日课堂内容已总结，请查看孩子作业。")

# ===== 题目提取与分类 =====
def extract_questions_by_knowledge(ocr_text, class_level):
    """让 Gemini 将识别出的文字整理成电子版题目，并按知识点分类"""
    prompt = f"""你是一位{ class_level }数学老师。请从以下OCR识别出的试卷文字中，提取每一道完整的题目，并按照知识点进行分类。
返回格式为 JSON，每个知识点下包含一个题目列表：
{{
  "知识点1": ["题目1完整内容", "题目2完整内容"],
  "知识点2": ["题目3完整内容"]
}}
如果OCR文字不清晰，请根据数学常见题型合理补全题目。只返回JSON，不要多余文字。

OCR 文字：
{ocr_text}
"""
    result_text = gemini_generate(prompt, fallback='{"未分类":["无法提取题目"]}')
    try:
        cleaned = re.sub(r'```json|```', '', result_text).strip()
        return json.loads(cleaned)
    except:
        st.error("题目分类解析失败，请稍后重试。")
        return {"未分类": ["解析异常"]}

# ===== 导出 Word 错题集 =====
def export_wrong_questions_to_docx(wrong_questions, class_level):
    """将错题列表导出为 Word 文档，返回文件路径"""
    doc = Document()
    doc.add_heading(f'{class_level} 错题集', 0)

    for q in wrong_questions:
        doc.add_paragraph(f"题号：{q.get('number', '未知')}", style='List Bullet')
        doc.add_paragraph(f"题目：{q.get('content', '')}")
        doc.add_paragraph(f"知识点：{q.get('knowledge', '未知')}")
        doc.add_paragraph("")  # 空行

    # 保存到临时文件
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    doc.save(tmp_file.name)
    return tmp_file.name

# ===== 页面初始化 =====
if "wrong_questions_list" not in st.session_state:
    st.session_state.wrong_questions_list = []   # 存储多张试卷的错题
if "test_summary" not in st.session_state:
    st.session_state.test_summary = ""           # 最近一次试卷的整体总结

st.set_page_config(page_title="数学教学助手 Pro", layout="wide")
st.title("📐 数学教学智能助手 Pro")
st.markdown("支持错题检测、专属错题集、课堂反馈生成、题目电子化。")

# ===== 功能选项卡 =====
tab1, tab2, tab3, tab4 = st.tabs(["📝 试卷分析与错题集", "📢 班级课堂反馈", "📋 题目电子化", "❓ 使用说明"])

# ==================== Tab1: 试卷分析与错题集 ====================
with tab1:
    st.header("上传已批改的试卷照片，自动识别错题并收录")
    class_level = st.selectbox("选择年级", ["七年级", "八年级", "九年级"], key="tab1_level")
    uploaded_files = st.file_uploader(
        "支持批量上传（每次可多选）", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🔍 开始分析试卷", type="primary"):
            all_wrong = []
            total_summaries = []
            progress = st.progress(0)
            for i, file in enumerate(uploaded_files):
                image = Image.open(file)
                st.image(image, caption=file.name, width=300)
                with st.spinner(f"正在识别 {file.name} ..."):
                    ocr_text = baidu_ocr(file.getvalue())
                    if ocr_text:
                        st.text_area(f"OCR 结果 - {file.name}", ocr_text, height=100)
                        analysis = analyze_test_paper(ocr_text, class_level)
                        if analysis:
                            wrong_qs = [q for q in analysis["questions"] if not q.get("correct", True)]
                            all_wrong.extend(wrong_qs)
                            total_summaries.append(analysis.get("summary", ""))
                            st.success(f"{file.name} 分析完成：错题 {len(wrong_qs)} 道")
                        else:
                            st.error(f"{file.name} 分析失败")
                    else:
                        st.warning(f"{file.name} 未识别到文字。")
                progress.progress((i+1)/len(uploaded_files))

            # 保存到 session_state
            if all_wrong:
                st.session_state.wrong_questions_list.extend(all_wrong)
                # 去重（简单按内容去重）
                seen = set()
                unique_wrong = []
                for q in st.session_state.wrong_questions_list:
                    key = q.get("content", "")
                    if key not in seen:
                        seen.add(key)
                        unique_wrong.append(q)
                st.session_state.wrong_questions_list = unique_wrong
                st.session_state.test_summary = "\n".join(total_summaries)
                st.success(f"共收录错题 {len(st.session_state.wrong_questions_list)} 道（已自动去重）")
            else:
                st.info("未检测到错题，或请检查图片是否包含对错标记。")

    # 显示已收录的错题
    if st.session_state.wrong_questions_list:
        st.subheader("📂 当前错题集")
        df = pd.DataFrame(st.session_state.wrong_questions_list)
        st.dataframe(df)

        # 知识点统计
        knowledge_counts = df['knowledge'].value_counts()
        st.bar_chart(knowledge_counts)

        # 生成家长反馈
        if st.button("💬 生成家长反馈"):
            with st.spinner("生成中..."):
                fb = parent_feedback(
                    st.session_state.wrong_questions_list,
                    knowledge_counts.index.tolist(),
                    class_level
                )
                st.text_area("家长反馈（可直接复制）", fb, height=200)

        # 导出错题集
        if st.button("⬇️ 导出错题集为 Word"):
            path = export_wrong_questions_to_docx(st.session_state.wrong_questions_list, class_level)
            with open(path, "rb") as f:
                st.download_button(
                    "下载错题集.docx",
                    f,
                    file_name="错题集.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            os.unlink(path)

        # 清空错题集
        if st.button("🗑️ 清空错题集"):
            st.session_state.wrong_questions_list = []
            st.session_state.test_summary = ""
            st.rerun()

# ==================== Tab2: 班级课堂反馈 ====================
with tab2:
    st.header("生成本节课的班级课堂反馈")
    st.markdown("结合之前试卷分析的整体情况（从错题集摘要自动获取）和本节课内容。")
    class_level_t2 = st.selectbox("年级", ["七年级", "八年级", "九年级"], key="tab2_level")

    # 显示已有测试总结
    if st.session_state.test_summary:
        with st.expander("📊 已有测试整体情况（自动从试卷分析获取）"):
            st.write(st.session_state.test_summary)
    else:
        st.info("请先在「试卷分析」Tab 中分析至少一张试卷，或手动输入测试整体情况。")

    # 本节课内容输入
    lesson_text = st.text_area("✏️ 输入本节课的文字描述（或补充说明）", height=100)
    lesson_image = st.file_uploader("📷 上传本节课板书/课件照片（可选）", type=["jpg","jpeg","png"])

    if st.button("📢 生成班级反馈", key="class_feedback_btn"):
        # 处理本节课图片文字
        extra_ocr = ""
        if lesson_image:
            with st.spinner("识别图片文字..."):
                extra_ocr = baidu_ocr(lesson_image.getvalue())
                st.text_area("图片识别结果", extra_ocr, height=80)
        full_lesson_desc = lesson_text + "\n" + extra_ocr

        test_summary = st.session_state.test_summary or "暂无测试数据"
        feedback = class_feedback(full_lesson_desc, test_summary, class_level_t2)
        st.text_area("课堂反馈（可复制到班群）", feedback, height=250)

# ==================== Tab3: 题目电子化 ====================
with tab3:
    st.header("将试卷/习题照片转换为电子版，并按知识点分类")
    class_level_t3 = st.selectbox("年级", ["七年级", "八年级", "九年级"], key="tab3_level")
    source_file = st.file_uploader("上传包含题目的图片或 PDF（暂只支持图片）", type=["jpg","jpeg","png"])

    if source_file and st.button("🔄 开始转换"):
        with st.spinner("OCR 识别中..."):
            ocr_text = baidu_ocr(source_file.getvalue())
        if ocr_text:
            st.text_area("OCR 文字", ocr_text, height=150)
            with st.spinner("Gemini 正在提取题目并分类..."):
                result_dict = extract_questions_by_knowledge(ocr_text, class_level_t3)
            st.subheader("📚 按知识点分类的电子版题目")
            for knowledge, qs in result_dict.items():
                with st.expander(f"**{knowledge}** ({len(qs)} 题)"):
                    for idx, q in enumerate(qs, 1):
                        st.markdown(f"{idx}. {q}")
        else:
            st.error("未识别到文字，请上传清晰的图片。")

# ==================== Tab4: 使用说明 ====================
with tab4:
    st.markdown("""
    ### 功能说明
    1. **试卷分析与错题集**  
       上传已批改的试卷照片（支持批量），程序自动识别√/×标记，提取错题并分析知识点。  
       错题会自动累积到“错题集”中，可导出 Word 文档，方便打印或存档。  
       点击“生成家长反馈”可获得一段可直接发送给家长的话。

    2. **班级课堂反馈**  
       输入本节课的文字描述或上传板书照片，程序会结合之前的试卷整体情况，  
       生成一段包含授课内容、重难点和学情分析的班群反馈。

    3. **题目电子化**  
       上传试卷或习题照片，程序提取每一道题并按照知识点分组显示，  
       方便您直接复制到课件或练习文档中。

    ### 注意事项
    - 请确保试卷照片中已经包含老师的批改痕迹（√ / × 或得分），以便准确判断对错。
    - 所有分析均依赖 Gemini 和百度 OCR，请确保网络畅通。
    - 错题集会暂存在当前浏览器会话中，关闭页面后数据会丢失，请及时导出。
    """)
