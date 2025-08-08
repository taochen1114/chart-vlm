import torch
import os
import csv
from PIL import Image
from transformers import AutoModel, AutoTokenizer
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

# ===== 📘 Function 說明總覽 =====

# 🔧 工具與編碼
# - stitch_images：將多張圖片橫向拼接為一張，用於模型推論。
# - encode_image：將圖片轉換為 base64 字串，供 GPT-4o API 使用。

# 🧠 GPT-4o 推論
# - openai_inference：將單張圖片與問題送入 GPT-4o，回傳文字答案。

# 🖼️ VLM 批次推論
# - run_vlm_on_all_images：將資料夾中的所有圖片逐張送入 VLM 並取得回答。

# 🧠 MiniCPM 推論邏輯
# - vlm_answer：將單張圖片與特定 prompt 送入 MiniCPM-V，取得回答。

# 🔍 篩選相關圖片頁面
# - filter_related_pages_by_vlm：逐頁詢問 VLM「這一頁是否與此問題有關」，回傳相關頁面。

# ❓ 對相關頁面問完整題目
# - run_vlm：對篩選出來的相關圖片頁面進行完整問題回答，回傳每頁的答案。

# 📚 整理結果
# - summarize_answers：從所有答案中選出最長回答當作代表答案，並整理頁碼。

# 📄 CSV 資料處理
# - read_column_as_list：從 CSV 中指定欄位讀取成 list，例如取得所有問題。

# ✅ 評估相關
# - exact_match_score：檢查預測短答案是否與標準答案完全一致（忽略格式差異）。
# - llm_judge_score：將長答案交給 GPT-4o 評分語意相似度（1~5 分）。
# - evaluate_with_llm：整合模型結果與評分邏輯，輸出完整評估結果為 CSV。

# 📌 補充：
# - bginfo：輔助 VLM 理解圖片格式的提示文字。
# - Prompt：根據不同問題類型提供的標準化 prompt 模板。



# ===== OpenAI gpt-4o api =====
def stitch_images(image_paths):
    images = [Image.open(p).convert("RGB").resize((720, 720)) for p in image_paths]
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    stitched = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in images:
        stitched.paste(img, (x_offset, 0))
        x_offset += img.width
    return stitched


def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def openai_inference(image_obj, question, OPENAI_API_KEY, MAX_NEW_TOKEN=256, TEMPERATURE=0.2, repetition_penalty=1.05):
    openai.api_key = OPENAI_API_KEY
    buffer = io.BytesIO()
    image_obj.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        max_tokens=MAX_NEW_TOKEN,
        temperature=TEMPERATURE,
        frequency_penalty=repetition_penalty
    )

    return response.choices[0].message.content.strip()



def run_vlm_on_all_images(image_folder, question, vlm_answer_fn, prompt_type, bginfo=None, resize=(720, 720)):
    results = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.lower().endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            img = Image.open(image_path).resize(resize)
            page_num = int(re.search(r'\d+', filename).group()) if re.search(r'\d+', filename) else filename

            answer = vlm_answer_fn(
                img, 
                question=question,
                prompt_type=prompt_type,
                bginfo=bginfo
            )

            results.append({
                "page": page_num,
                "image_path": image_path,
                "answer": answer
            })

    print(f"✅ 所有圖片已完成 VLM 推論，共處理 {len(results)} 頁")
    return results


# ===== 初始化模型 =====
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-V-2_6',
    trust_remote_code=True,
    attn_implementation='sdpa',
    torch_dtype=torch.bfloat16
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

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
          
          "Compare": '''請你根據圖片中的數據圖表來回答問題。如果你能從圖表中找到資訊，請給出明確且具體的答案。

            例如：「1Q25 與 1Q24 相比，手續費淨收益總額成長了多少％？」  
            答：33%。若無單位請務必加上單位。

            請用繁體中文回答，並使用以下 JSON 格式輸出：

            {
            "short_answer": "33%",
            "long_answer": "1Q25 與 1Q24 相比，手續費淨收益總額成長了 33%。",
            "page": [頁碼數字]
            }

            請注意：
            - short_answer：僅包含數值與單位（如 "33%"、"2.6億元"）
            - long_answer：完整說明比較結果，包含時間、指標與變化內容
            - page：整數陣列，代表答案出處的頁碼；若有多頁，請依照數字大小排序

            若圖表出現在多頁，請使用此格式：
            {
            "short_answer": "33%",
            "long_answer": "1Q25 與 1Q24 相比，手續費淨收益總額成長了 33%。",
            "page": [5, 12]
            }

            若無法從圖表中取得資訊，請輸出：
            {
            "short_answer": "無法取得",
            "long_answer": "無法從圖表中找到相關資訊",
            "page": []
            }
          ''',
          
          "Rank":'''請你根據圖片中的數據圖表來回答問題。如果你能從圖表中找到資訊，請依照指標數值的高低進行排序，並輸出排序結果。

            請用繁體中文回答，並使用以下 JSON 格式輸出：

            {
            "short_answer": "排序結果（例如：1Q21 > 1Q22 > 1Q24 > 1Q25 > 1Q23）",
            "long_answer": "詳列各項排序及數值（例如：1Q21（6.31%） > 1Q22（4.74%） > 1Q24（4.39%） > 1Q25（4.01%） > 1Q23（2.69%））",
            "page": [頁碼數字]
            }

            請注意：
            - `short_answer` 為純排序，僅保留項目順序，省略數值
            - `long_answer` 為完整排序，需列出每個項目的對應數值
            - `page` 為整數陣列，指出圖表出現的頁碼，若有多頁請排序列出

            若無法從圖表中取得資料，請回答：

            {
            "short_answer": "無法取得",
            "long_answer": "無法從圖表中找到相關資訊",
            "page": []
            }

            以下為範例：

            問題：「請依照避險後投資收益率的高低排序 1Q21～1Q25 各季資料。」

            回答：
            {
            "short_answer": "1Q21 > 1Q22 > 1Q24 > 1Q25 > 1Q23",
            "long_answer": "1Q21（6.31%） > 1Q22（4.74%） > 1Q24（4.39%） > 1Q25（4.01%） > 1Q23（2.69%）",
            "page": [25]
            }
            ''',

          "Trend": '''請你根據圖片中的數據圖表來回答問題。如果你能從圖表中找到資訊，請給出明確且具體的答案。

            當問題詢問某指標在一段期間內的變化趨勢時，請根據實際數據情況簡潔回答：
            - 若數值明顯上升，請回答「上升」
            - 若數值明顯下降，請回答「下滑」或「下降」
            - 若數值變化不大，請回答「持平」
            - 若變化有轉折，請如實回答「先升後降」、「先降後升」等

            請用繁體中文回答，並使用以下 JSON 格式輸出：

            {
            "short_answer": "趨勢簡述（例如：下滑）",
            "long_answer": "具體說明趨勢與數值變化（例如：持續下滑，從 45.9%（FY20）一路下降至 8.9%（FY23））",
            "page": [頁碼數字]
            }

            若出現多頁，請依數字大小排序：

            {
            "short_answer": "下滑",
            "long_answer": "持續下滑，從 45.9%（FY20）一路下降至 8.9%（FY23）",
            "page": [10, 11]
            }

            若無法從圖表中取得資料，請回答：

            {
            "short_answer": "無法取得",
            "long_answer": "無法從圖表中找到相關資訊",
            "page": []
            }

            範例題目與回答：

            問題：「海外獲利佔比在 FY20 至 FY23 呈現什麼趨勢？」

            回答：
            {
            "short_answer": "下滑",
            "long_answer": "持續下滑，從 45.9%（FY20）一路下降至 8.9%（FY23）",
            "page": [10]
            }
          ''',
          
          "Extract":'''請你根據圖片中的數據圖表來回答問題。如果你能從圖表中找到資訊，請給出明確且具體的答案。

            請以最簡單的方式回答：  
            - 若答案為單一項目，請直接回答項目名稱或數值  
            - 若需列出多個項目，請依照以下規則分隔：  
            - 中文項目：使用頓號（、）  
            - 英文項目：使用逗號（,）

            請用繁體中文回答，並使用以下 JSON 格式輸出：

            {
            "short_answer": "擷取出的項目（最簡格式，使用正確分隔符號）",
            "long_answer": "包含上下文說明的完整回答",
            "page": [頁碼數字]
            }

            若出現多頁，請依頁碼大小排序，例如：

            {
            "short_answer": "信用卡、財富管理、外匯管理、聯貸、其他",
            "long_answer": "手續費淨收益的構成項目包括：信用卡、財富管理、外匯管理、聯貸及其他。",
            "page": [9, 11]
            }

            若無法從圖表中取得資訊，請回答：

            {
            "short_answer": "無法取得",
            "long_answer": "無法從圖表中找到相關資訊",
            "page": []
            }
            '''}
bginfo = "大標題通嘗在左上角，頁碼位於右下角。長條圖的標題位於圖片正上方。"

# ===== VLM 回答函式 =====
def vlm_answer(image: Image.Image, question: str, prompt_type: str) -> str:
    prompt = Prompt[prompt_type] + f"\n\n{bginfo}\n\n"
    full_question = f"{question}\n\n{prompt}"
    imgs = [image]
    msgs = [{'role': 'user', 'content': imgs + [full_question]}]

    with torch.no_grad():
        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.2,
            max_new_tokens=256
        )
    return answer

def run_vlm_multi_images(matched_pages, question, vlm_answer_fn, prompt_type, resize=(720, 720)):
    if not matched_pages:
        return []

    # 把多頁圖片一次送進模型
    imgs = [Image.open(p['image_path']).resize(resize) for p in matched_pages]
    stitched_img = stitch_images([p['image_path'] for p in matched_pages])
    answer = vlm_answer_fn(stitched_img, question, prompt_type)
    print(f"📄 多頁合併回答：{answer}")

    return [{"page": p['page'], "image_path": p['image_path'], "answer": answer} for p in matched_pages]


# ===== 篩選相關頁面 =====
def filter_related_pages_by_vlm(image_folder, question, vlm_answer_fn, resize=(720, 720)):
    related_pages = []
    
    positive_keywords = ["是", "相關", "提到", "顯示", "包含"]
    negative_keywords = ["否", "不相關", "無關", "無法取得", "不包含", "沒有提到", "不清楚"]

    for filename in sorted(os.listdir(image_folder)):
        if filename.lower().endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            img = Image.open(image_path).resize(resize)

            # 解析頁碼
            match = re.search(r'\d+', filename)
            page_num = int(match.group()) if match else filename

            # 建立 prompt
            relevance_question = (
                f"請你判斷這一頁是否與以下問題有關：「{question}」。"
                "如果有關，請回答「是」；如果無關或無法判斷，請回答「否」。"
            )
            
            # 發送給 VLM
            answer = vlm_answer_fn(img, relevance_question, prompt_type="Trend").strip()
            print(f"🧐 頁 {page_num} → 回答：「{answer}」")

            # 判斷是否為相關頁
            answer_lower = answer.lower()
            has_positive = any(kw in answer_lower for kw in positive_keywords)
            has_negative = any(kw in answer_lower for kw in negative_keywords)

            if has_positive and not has_negative:
                related_pages.append({"page": page_num, "image_path": image_path})
    
    return related_pages

# ===== 執行 VLM 問完整問題 =====
def run_vlm(matched_pages, question, vlm_answer_fn, prompt_type, resize=(720, 720)):
    results = []
    for page_info in matched_pages:
        img = Image.open(page_info['image_path']).resize(resize)
        answer = vlm_answer_fn(img, question, prompt_type)
        print(f"📄 第 {page_info['page']} 頁 → 回答：{answer}")
        results.append({"page": page_info['page'], "image_path": page_info['image_path'], "answer": answer})
    return results


# ===== 整理最終答案 =====
def summarize_answers(results):
    if not results:
        return {"final_answer": "找不到任何相關資訊", "pages": []}
    best = max(results, key=lambda r: len(r["answer"]))
    return {"final_answer": best["answer"], "pages": [r["page"] for r in results]}


def read_column_as_list(csv_path: str, column_name: str) -> list:
    result = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if column_name not in reader.fieldnames:
            raise ValueError(f"❌ 找不到欄位：{column_name}")
        for row in reader:
            value = row[column_name].strip()
            if value:
                result.append(value)
    return result

def exact_match_score(prediction: str, reference: str) -> int:
    """
    Returns 1 if prediction exactly matches reference (ignoring space and character form), else 0.
    """
    def normalize(text):
        text = unicodedata.normalize("NFKC", text)
        return re.sub(r"\s+", "", text.strip())
    return int(normalize(prediction) == normalize(reference))

def llm_judge_score(question: str, reference: str, prediction: str, model_name: str, openai_api_key: str) -> int:
    openai.api_key = openai_api_key

    prompt = f"""
        你是一位專業的 AI 評估助理，請根據以下問題、標準答案與模型回覆，給出 1～5 的相似度分數。
        1：完全不相關，5：完全一致。

        ---
        問題：
        {question}

        標準答案：
        {reference}

        模型（{model_name}）的回答：
        {prediction}

        請你只回答一個整數分數（1–5），不要附加任何說明。
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=10
        )
        score = int(re.search(r"[1-5]", response.choices[0].message.content).group())
        return score
    except Exception as e:
        print(f"⚠️ GPT 評分錯誤：{e}")
        return -1

def evaluate_with_llm(csv_path, model_results, model_name, openai_api_key, output_csv="final_eval_output.csv"):
    df = pd.read_csv(csv_path)
    results = []

    for item in model_results:
        question = item["question"]
        raw = item["answer"]

        try:
            pred_json = eval(raw)
            short_pred = pred_json.get("short_answer", raw)
            long_pred = pred_json.get("long_answer", raw)
        except:
            short_pred = raw
            long_pred = raw

        ref_row = df[df["Question"] == question]
        if ref_row.empty:
            print(f"❌ 找不到對應題目：{question}")
            continue

        ref_data = ref_row.iloc[0]
        short_ref = ref_data["Short Answer"]
        long_ref = ref_data["Long Answer"]
        q_type = ref_data.get("Chart Type", "Unknown")

        em = exact_match_score(short_pred, short_ref)
        llm_score = llm_judge_score(question, long_ref, long_pred, model_name, openai_api_key)

        results.append({
            "Type": q_type,
            "Question": question,
            "Short Answer": short_ref,
            "Long Answer": long_ref,
            "Prediction": long_pred,
            "Exact Match Score": em,
            "LLM Judge Score": llm_score,
            "Model": model_name
        })

    final_df = pd.DataFrame(results)
    final_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 評估完成，已輸出至：{output_csv}")
    return final_df





if __name__ == '__main__':
    image_folder = "./data/pdf_img/1Q25_MP_Chinese_Vupload/"
    csv_path = "./QA/國泰法說會簡報QA - 定位問題.csv"
    question_col = "Question"
    prompt_type = "Locate"  # Compare, Extract, Locate, Rank, Trend

    questions = read_column_as_list(csv_path, question_col)
    all_model_results = []

    for idx, question in enumerate(questions, 1):
        print(f"\n🟢 問題 {idx}/{len(questions)}：{question}")

        # 1. 篩選相關頁
        related_pages = filter_related_pages_by_vlm(
            image_folder=image_folder,
            question=question,
            vlm_answer_fn=vlm_answer
        )

        # 2. 根據題型決定如何問
        if prompt_type == "Locate":
            # 多頁合併送進模型
            results = run_vlm_multi_images(
                matched_pages=related_pages,
                question=question,
                vlm_answer_fn=vlm_answer,
                prompt_type=prompt_type
            )
        else:
            # 原本的逐頁方式
            results = run_vlm(
                matched_pages=related_pages,
                question=question,
                vlm_answer_fn=vlm_answer,
                prompt_type=prompt_type
            )

        # 3. 整理答案
        final = summarize_answers(results)
        print("\n🎯 最終答案：")
        print(final)

        all_model_results.append({
            "question": question,
            "answer": final["final_answer"]
        })

    # 4. 評估
    evaluate_with_llm(
        csv_path=csv_path,
        model_results=all_model_results,
        model_name="MiniCPM-V-2_6",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        output_csv="vlm_eval_with_llm_Locate.csv"
    )


