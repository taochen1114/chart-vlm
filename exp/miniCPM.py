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
import argparse


# def openai_inference(image_path, model_key, OPENAI_API_KEY ,question_key ,ground_text=None, MAX_NEW_TOKEN=128, TEMPERATURE=0, repetition_penalty=1.05):
#     client = OpenAI(api_key=OPENAI_API_KEY)
#     model = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS[model_key])
#     image_base64 = encode_image(image_path)
#     question = generate_text(question_key, ground_text)[0]
#     response = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {
#                     "role": "user",
#                     "content": [
#                                     {"type": "text", "text": question},
#                                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
#                                 ],
#                     }
#                 ],
#                 max_tokens=MAX_NEW_TOKEN,
#                 temperature=TEMPERATURE,
#                 frequency_penalty=repetition_penalty,
#                 )
#     output_text = [response.choices[0].message.content]
#     prediction = process_and_clean_text(output_text)
#     return prediction

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

Prompt = {"Locate": "請你根據圖片中的數據圖表來回答問題。如果你能從圖片中找到資訊，請給出明確且具體的答案"
          "請簡答，例如：「xxx在第幾頁？」或「xxx在哪裡」請回答 「第y頁」或有超過一頁可以回答「第y頁和、第z頁」"
          "繁體中文回答問題",
          
          "Compare": "請你根據圖片中的數據圖表來回答問題。如果你能從圖片中找到資訊，請給出明確且具體的答案"
          "請簡答，例如:問題問數字相關的問題並且有單位，請直接回答數字即可。若無單位請務必加上單位。"
          "繁體中文回答",
          
          "Rank":"請你根據圖片中的數據圖表來回答問題。如果你能從圖片中找到資訊，請給出明確且具體的答案"
          "如果遇到最高或是最低請直接以文字回答最高或是最低的項目，如果是請以高低排序請以大於（>）或是小於（<）符號表示。例如：「A>B>C」，反之亦然。"
          "繁體中文回答問題",

          "Trend": "請你根據圖片中的數據圖表來回答問題。如果你能從圖片中找到資訊，請給出明確且具體的答案"
          "問趨勢或是變化時可以上升下降或是持平為答案，但有時會出現先升後降或是先將後升，請以實際情況為主。"
          "繁體中文回答問題。",
          
          "Extract":"請你根據圖片中的數據圖表來回答問題。如果你能從圖片中找到資訊，請給出明確且具體的答案"
          "請以最間單的方式回答，如需列出多個項目，若為中文請以頓號（、）分隔，若為英文請以逗號（,）分隔。"
          "繁體中文回答問題。"}

# ===== 單次提問（附加失敗提示）=====
def ask(images, question , prompt_type="Locate"):
    prompt = Prompt[prompt_type]
    full_question = f"{question}\n\n{prompt}"
    msgs = [{'role': 'user', 'content': images + [full_question]}]
    with torch.no_grad():
        answer = model.chat(
            image=None, 
            msgs=msgs, 
            tokenizer=tokenizer,
            sampling=True,             # 使回答可重現，避免胡亂生成
            temperature=0.2,            # 趨近 deterministic，適合 QA 評估任務
            max_new_tokens=256)          # 回答不夠完整時可調大（如 512）)
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
    return int(any(word in pred for word in ref_keywords))


def evaluate_answer(pred, ref):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_score = scorer.score(ref, pred)['rougeL'].fmeasure
    hit_score = substring_hit(pred, ref)
    return {
        "rougeL_f1": round(rouge_l_score, 4),
        "substring_hit": hit_score,
    }

# ===== 處理 CSV 問題 =====
def process_csv_questions(csv_path, image_folder, batch_size):
    results = []

    # 預先載入所有圖片批次
    image_batches = list(load_image_batches(image_folder, batch_size=batch_size))

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            question = row['Question']
            reference = row['Short Answer']
            print(f"\n============= 問題 {idx}: {question} =============")

            summaries = []
            for i, images in enumerate(image_batches, 1):
                try:
                    summary = ask(images, question)
                except Exception as e:
                    print(f"  ❌ 圖片批次 {i} 發生錯誤: {e}")
                    summary = "(ERROR)"
                summaries.append(summary)

            final_answer = summarize_all(summaries, question)
            print(f"✅ 最終回答：{final_answer}")

            metrics = evaluate_answer(final_answer, reference)
            results.append({
                "id": idx,
                "question": question,
                "prediction": final_answer,
                "reference": reference,
                **metrics
            })

    return results

# ===== 主流程 =====
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default='./QA/國泰法說會簡報QA - 資料提取問題.csv')
    parser.add_argument("--folder", type=str, default='./output_images_fitz')
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    # csv_path = './QA/國泰法說會簡報QA - 資料提取問題.csv'
    # folder = './output_images_fitz'
    csv_path = args.csv_path
    folder = args.folder
    batch_size = args.batch_size


    results = process_csv_questions(csv_path, folder, batch_size=batch_size)

    # 統計
    total_rouge = 0
    total_hit = 0
    count = len(results)

    print("\n📋 每一題評估結果：")
    for r in results:
        rouge = r.get('rougeL_f1', 0)
        hit = r.get('substring_hit', 0)
        print(f"ID {r['id']} | ROUGE-L F1: {rouge} | Substring Hit: {hit}")

        total_rouge += rouge
        total_hit += hit

    avg_rouge = total_rouge / count if count > 0 else 0
    avg_hit = total_hit / count if count > 0 else 0

    print("\n📊 Evaluation Summary:")
    print(f"🔹 Average ROUGE-L F1: {round(avg_rouge, 4)}")
    print(f"🔹 Average Substring Hit Rate: {round(avg_hit, 4)}")
