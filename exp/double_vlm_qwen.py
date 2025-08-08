import torch
import os
import csv
from PIL import Image
from rouge_score import rouge_scorer
from dotenv import load_dotenv
load_dotenv()
import re
import unicodedata
import openai
import base64
import io
import time
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json

# ===== 初始化模型 =====
model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_name_or_path)

# ===== Prompt 定義 =====
Prompt = {
    "Locate": '''請你根據圖片中的數據圖表來回答問題。如果你能從圖片中找到資訊，請給出明確且具體的答案。
    例如：「文件中『每股盈餘』的數據圖出現在簡報的哪一頁？」
    請用繁體中文回答，並使用以下 JSON 格式輸出：
    {
      "answer": "詳答內容（例如：『每股盈餘』的數據圖出現在第5頁）",
      "page": [頁碼數字]
    }
    如果圖表出現在多頁，請用整數陣列標示所有頁碼，例如：
    {
      "answer": "『每股盈餘』的數據圖出現在第5頁和第12頁",
      "page": [5, 12]
    }
    如果無法找到相關資訊，請回答：
    {
      "answer": "無法從圖表中找到『每股盈餘』的資訊",
      "page": []
    }''',
    "Compare": '''…（略，同上，保留原 prompt 內容）…''',
    "Rank":   '''…''',
    "Trend":  '''…''',
    "Extract":'''…''',
    "Relevance": '''判斷這張圖片的內容是否與以下問題有關：
    「{question}」
    請務必只輸出一個字：
    - 如果有關，請回答：是
    - 如果無關或無法判斷，請回答：否
    '''
}
bginfo = "大標題通嘗在左上角，頁碼位於右下角。長條圖的標題位於圖片正上方。"

# ===== VLM 推論函式 =====
def vlm_answer(image: Image.Image, question: str, prompt_type: str) -> str:
    if prompt_type == "Relevance":
        prompt = Prompt["Relevance"].format(question=question)
        max_tokens = 5
    else:
        prompt = Prompt[prompt_type] + f"\n\n{bginfo}\n\n"
        prompt = f"{question}\n\n{prompt}"
        max_tokens = 256

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    gen_ids = model.generate(**inputs, max_new_tokens=max_tokens)
    output = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # 清掉多餘模板，只保留最後一行
    output = output.split("\n")[-1].strip()

    # 如果是單個 } 或 { 或空字串，視為無效
    if output in ["}", "{", ""]:
        return '{"short_answer": "無法取得", "long_answer": "無法從圖表中找到相關資訊", "page": []}'

    return output

# ===== 篩選相關頁面 =====
def filter_related_pages_by_vlm(image_folder, question, vlm_answer_fn, resize=(720, 720)):
    related_pages = []
    for fn in sorted(os.listdir(image_folder)):
        if fn.lower().endswith(".png"):
            path = os.path.join(image_folder, fn)
            img = Image.open(path).resize(resize)
            match = re.search(r'\d+', fn)
            page_num = int(match.group()) if match else fn

            answer = vlm_answer_fn(img, question, "Relevance").strip()
            print(f"🧐 頁 {page_num} → 回答：「{answer}」")

            if "是" in answer and "否" not in answer:
                related_pages.append({"page": page_num, "image_path": path})

    return related_pages

# ===== 執行 VLM 問完整問題 =====
def run_vlm(matched_pages, question, vlm_answer_fn, prompt_type, resize=(720, 720)):
    results = []
    for info in matched_pages:
        img = Image.open(info['image_path']).resize(resize)
        answer = vlm_answer_fn(img, question, prompt_type)
        print(f"📄 第 {info['page']} 頁 → 原始輸出：{repr(answer)}")
        results.append({"page": info['page'], "answer": answer})
    return results

# ===== 整理最終答案（已過濾無效回答） =====
def summarize_answers(results):
    # 過濾掉所有「無法取得」
    valid_results = [r for r in results if '"無法取得"' not in r["answer"]]
    if not valid_results:
        return {"final_answer": "找不到任何相關資訊", "pages": []}
    # 選最長的答案
    best = max(valid_results, key=lambda r: len(r["answer"]))
    return {
        "final_answer": best["answer"],
        "pages": [r["page"] for r in valid_results]
    }

# ===== CSV 工具 =====
def read_column_as_list(csv_path: str, column_name: str) -> list:
    vals = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if column_name not in reader.fieldnames:
            raise ValueError(f"❌ 找不到欄位：{column_name}")
        for row in reader:
            v = row[column_name].strip()
            if v:
                vals.append(v)
    return vals

# ===== 評估工具 =====
def exact_match_score(pred: str, ref: str) -> int:
    def normalize(txt):
        txt = unicodedata.normalize("NFKC", txt)
        return re.sub(r"\s+", "", txt.strip())
    return int(normalize(pred) == normalize(ref))

def llm_judge_score(question, reference, prediction, model_name, openai_api_key) -> int:
    openai.api_key = openai_api_key
    prompt = f"""
你是一位專業的 AI 評估助理，請根據以下內容給出 1～5 分。
---
問題：
{question}
標準答案：
{reference}
模型答案：
{prediction}
只回一個整數（1–5）。
"""
    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10
        )
        return int(re.search(r"[1-5]", resp.choices[0].message.content).group())
    except Exception as e:
        print(f"⚠️ 評分錯誤：{e}")
        return -1

def safe_json_load(raw):
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group()) if m else None
    except Exception as e:
        print(f"⚠️ JSON 解析錯誤：{e}")
        return None

def evaluate_with_llm(csv_path, model_results, model_name, openai_api_key, output_csv="final_eval_output.csv"):
    df = pd.read_csv(csv_path)
    recs = []
    for itm in model_results:
        q = itm["question"]
        raw = itm["answer"]
        js = safe_json_load(raw)
        short_pred = js.get("short_answer", raw) if js else raw
        long_pred  = js.get("long_answer", raw)  if js else raw

        row = df[df["Question"] == q]
        if row.empty:
            print(f"❌ 找不到題目：{q}")
            continue
        sr = row.iloc[0]
        em = exact_match_score(short_pred, sr["Short Answer"])
        lj = llm_judge_score(q, sr["Long Answer"], long_pred, model_name, openai_api_key)
        recs.append({
            "Type": sr.get("Chart Type",""),
            "Question": q,
            "Short Answer": sr["Short Answer"],
            "Long Answer": sr["Long Answer"],
            "Prediction": long_pred,
            "Exact Match Score": em,
            "LLM Judge Score": lj,
            "Model": model_name
        })

    final_df = pd.DataFrame(recs)
    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 評估完成，已輸出：{output_csv}")
    return final_df

# ===== 主程式 =====
if __name__ == '__main__':
    image_folder = "./data/pdf_img/1Q25_MP_Chinese_Vupload/"
    csv_path     = "./QA/國泰法說會簡報QA - 定位問題.csv"
    questions    = read_column_as_list(csv_path, "Question")
    all_results  = []

    for idx, q in enumerate(questions, 1):
        print(f"\n🟢 問題 {idx}/{len(questions)}：{q}")
        # 1. 篩選相關頁
        pages = filter_related_pages_by_vlm(image_folder, q, vlm_answer)
        # 2. 正式回答
        res = run_vlm(pages, q, vlm_answer, prompt_type="Locate")
        final = summarize_answers(res)
        print("🎯 最終答案：", final)
        all_results.append({"question": q, "answer": final["final_answer"]})

    # 3. 評估
    evaluate_with_llm(
        csv_path, all_results,
        model_name="Qwen2.5-VL-7B-Instruct",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        output_csv="qwen_vlm_eval_with_llm_Locate.csv"
    )
