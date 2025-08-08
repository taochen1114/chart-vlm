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


# ===== 初始化模型 =====
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-V-2_6',
    trust_remote_code=True,
    attn_implementation='sdpa',
    torch_dtype=torch.bfloat16
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

# ===== 載入圖片批次 =====
def load_image_batches(folder_path, batch_size):
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    for i in range(0, len(image_files), batch_size):
        paths = image_files[i:i + batch_size]
        images = [Image.open(p).convert("RGB").resize((720, 720)) for p in paths]
        yield images


Prompt = {"Locate": '''請你根據圖片中的數據圖表來回答問題。如果你能從圖片中找到資訊，請給出明確且具體的答案。
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

# ===== 單次提問（附加失敗提示）=====
def ask(images, question , prompt_type="Extract"):
    prompt = Prompt[prompt_type]
    full_question = f"{question}\n\n{prompt}"
    msgs = [{'role': 'user', 'content': images + [full_question]}]
    with torch.no_grad():
        answer = model.chat(
            image=None, 
            msgs=msgs, 
            tokenizer=tokenizer,
            sampling=True,             
            temperature=0.2,            
            max_new_tokens=256)          
    return answer

# ===== 整合摘要再次提問 =====
def summarize_all(summaries, question):
    combined = "\n".join([f"批次{i+1}：{s}" for i, s in enumerate(summaries)])
    final_prompt = f"以下是每批圖片的摘要：\n{combined}\n\n請根據以下每批圖片的摘要內容，找出最合理的具體答案，只需直接回答結果即可，敬請使用繁體中文回答：{question}"
    return ask([], final_prompt)

# ===== 評估指標 =====
def substring_hit(pred, ref):
    pred = pred.lower().strip()
    ref = ref.lower().strip()
    ref_keywords = ref.split()
    
    for word in ref_keywords:
        if re.search(r'\b{}\b'.format(re.escape(word)), pred):
            return 1
    return 0


def evaluate_answer(pred, ref):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_score = scorer.score(ref, pred)['rougeL'].fmeasure
    hit_score = substring_hit(pred, ref)
    return {
        "rougeL_f1": round(rouge_l_score, 4),
        "substring_hit": hit_score,
    }

# ===== 處理 CSV 問題 =====
def process_csv_questions(csv_path, image_folder, OPENAI_API_KEY, batch_size=3):
    results = []

    # 預先載入所有圖片檔名
    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    # 批次化圖片
    def get_batches():
        for i in range(0, len(image_files), batch_size):
            yield image_files[i:i + batch_size]

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            question = row['Question']
            reference = row['Short Answer']
            print(f"\n============= 問題 {idx}: {question} =============")

            mcp_summaries = []
            openai_summaries = []

            # 每題所有 batch 都過 MiniCPM 和 OpenAI
            for i, image_batch_paths in enumerate(get_batches(), 1):
                # 👉 MiniCPM 多圖處理
                try:
                    images = [Image.open(p).convert("RGB").resize((720, 720)) for p in image_batch_paths]
                    mcp_summary = ask(images, question)
                except Exception as e:
                    print(f"❌ MiniCPM 圖片批次 {i} 發生錯誤: {e}")
                    mcp_summary = "(ERROR)"
                mcp_summaries.append(mcp_summary)

                # 👉 OpenAI 拼圖後處理
                try:
                    stitched_img = stitch_images(image_batch_paths)
                    openai_summary = openai_inference(stitched_img, question, OPENAI_API_KEY)
                except Exception as e:
                    print(f"❌ OpenAI 圖片批次 {i} 發生錯誤: {e}")
                    openai_summary = "(ERROR)"
                openai_summaries.append(openai_summary)

            # 整合摘要提問一次
            final_mcp = summarize_all(mcp_summaries, question)
            final_openai = summarize_all(openai_summaries, question)

            # 評估
            mcp_metrics = evaluate_answer(final_mcp, reference)
            openai_metrics = evaluate_answer(final_openai, reference)

            print(f"🟢 MiniCPM 回答：{final_mcp}")
            print(f"🔵 OpenAI 回答：{final_openai}")
            print(f"➡️ ROUGE MiniCPM: {mcp_metrics['rougeL_f1']} | OpenAI: {openai_metrics['rougeL_f1']}")
            print(f"➡️ Hit MiniCPM: {mcp_metrics['substring_hit']} | OpenAI: {openai_metrics['substring_hit']}")

            results.append({
                "id": idx,
                "question": question,
                "reference": reference,
                "mcp_prediction": final_mcp,
                "openai_prediction": final_openai,
                **{
                    "mcp_rouge": mcp_metrics['rougeL_f1'],
                    "mcp_hit": mcp_metrics['substring_hit'],
                    "openai_rouge": openai_metrics['rougeL_f1'],
                    "openai_hit": openai_metrics['substring_hit']
                }
            })

    return results


# ===== 主流程 =====
if __name__ == '__main__':
    start_time = time.time()
    csv_path = './QA/國泰法說會簡報QA - 資料提取問題.csv'
    folder = './output_images_fitz'
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    results = process_csv_questions(csv_path, folder, OPENAI_API_KEY, batch_size=3)

    # 統計
    total_rouge_mcp = 0
    total_hit_mcp = 0
    total_rouge_openai = 0
    total_hit_openai = 0
    count = len(results)

    print("\n📋 每一題評估結果：")
    for r in results:
        total_rouge_mcp += r['mcp_rouge']
        total_hit_mcp += r['mcp_hit']
        total_rouge_openai += r['openai_rouge']
        total_hit_openai += r['openai_hit']

    # 平均
    def safe_avg(total, count):
        return round(total / count, 4) if count > 0 else 0.0

    print("\n📊 評估總結：")
    print(f"🟢 MiniCPM - 平均 ROUGE-L F1: {safe_avg(total_rouge_mcp, count)}")
    print(f"🟢 MiniCPM - 平均 Substring Hit: {safe_avg(total_hit_mcp, count)}")
    print(f"🔵 OpenAI - 平均 ROUGE-L F1: {safe_avg(total_rouge_openai, count)}")
    print(f"🔵 OpenAI - 平均 Substring Hit: {safe_avg(total_hit_openai, count)}")
    
    end_time = time.time()  # 結束計時
    total_time = round(end_time - start_time, 2)
    minutes, seconds = divmod(total_time, 60)
    print(f"\n⏱️ 總執行時間：{int(minutes)} 分 {int(seconds)} 秒")



