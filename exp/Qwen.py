import os
import csv
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from rouge_score import rouge_scorer
import torch
import time
import io
import base64
import openai


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


Prompt = {
    "Locate": (
        "你是專業的金融人員，請你擷取圖片中的數據圖表來回答問題。如果你能從圖片中找到資訊，請給出明確且具體的答案"
        "請簡答，例如：「xxx在第幾頁？」或「xxx在哪裡」請回答 「第y頁」或有超過一頁可以回答「第y頁和、第z頁」"
        "繁體中文回答問題"
    ),
    "Compare": (
        "你是專業的金融人員，請你擷取圖片中的數據圖表來回答問題。如果你能從圖片中找到資訊，請給出明確且具體的答案"
        "請簡答，例如:問題問數字相關的問題並且有單位，請直接回答數字即可。若無單位請務必加上單位。"
        "繁體中文回答"
    ),
    "Rank": (
        "你是專業的金融人員，請你擷取圖片中的數據圖表來回答問題。如果你能從圖片中找到資訊，請給出明確且具體的答案"
        "如果遇到最高或是最低請直接以文字回答最高或是最低的項目，如果是請以高低排序請以大於（>）或是小於（<）符號表示。例如：「A>B>C」，反之亦然。"
        "繁體中文回答問題"
    ),
    "Trend": (
        "你是專業的金融人員，請你擷取圖片中的數據圖表來回答問題。如果你能從圖片中找到資訊，請給出明確且具體的答案"
        "問趨勢或是變化時可以上升下降或是持平為答案，但有時會出現先升後降或是先將後升，請以實際情況為主。"
        "繁體中文回答問題。"
    ),
    "Extract": (
        "你是專業的金融人員，請你擷取圖片中的數據圖表來回答問題。如果你能從圖片中找到資訊，請給出明確且具體的答案"
        "請以最簡單的方式回答，如需列出多個項目，若為中文請以頓號（、）分隔，若為英文請以逗號（,）分隔。"
        "繁體中文回答問題。"
    ),
}


def run_openai_batch(image_files, question, batch_size, OPENAI_API_KEY, prompt_type="Locate"):
    openai.api_key = OPENAI_API_KEY
    full_question = f"{question}\n\n{Prompt[prompt_type]}"
    summaries = []

    for i in range(0, len(image_files), batch_size):
        batch_paths = image_files[i:i + batch_size]
        stitched_image = stitch_images(batch_paths)

        buffer = io.BytesIO()
        stitched_image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_question},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    }
                ],
                max_tokens=256,
                temperature=0.2
            )
            summaries.append(response.choices[0].message.content.strip())
        except Exception as e:
            summaries.append(f"(ERROR: {e})")

    return summaries


def run_vlm_batch(image_files, question, batch_size, processor, model, prompt_type="Rank"):
    full_question = f"{question}\n\n{Prompt[prompt_type]}"
    summaries = []
    for i in range(0, len(image_files), batch_size):
        batch_paths = image_files[i:i + batch_size]
        images = [Image.open(p).convert("RGB").resize((720, 720)) for p in batch_paths]

        message = [{
            "role": "user",
            "content": ([{"type": "image", "image": img} for img in images] +
                        [{"type": "text", "text": full_question}])
        }]

        try:
            text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(message)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True
            ).to("cuda")

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.2)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, outputs)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            summaries.append(decoded.strip())
        except torch.cuda.OutOfMemoryError:
            summaries.append("（OOM）")
            torch.cuda.empty_cache()
    return summaries


def summarize_final_answer(summaries, question, processor, model):
    combined = "\n".join([f"批次 {i+1}：{s}" for i, s in enumerate(summaries)])
    final_prompt = f"以下是每批圖片的摘要：\n{combined}\n\n請根據以下每批圖片的摘要內容，找出最合理的具體答案，只需直接回答結果即可，敬請使用繁體中文回答：{question}"

    message = [{"role": "user", "content": [{"type": "text", "text": final_prompt}]}]
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.2)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def evaluate_answer(pred, ref):
    pred, ref = pred.lower().strip(), ref.lower().strip()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = scorer.score(ref, pred)['rougeL'].fmeasure
    hit = int(any(word in pred for word in ref.split()))
    return {"rougeL_f1": round(rouge, 4), "substring_hit": hit}


def process_csv_questions(csv_path, image_folder, processor, model, OPENAI_API_KEY, batch_size):
    results = []

    image_files = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, 1):
            question = row['Question']
            reference = row['Short Answer']
            print(f"\n============= 問題 {idx}: {question} =============")

            qwen_summaries = run_vlm_batch(image_files, question, batch_size, processor, model)
            qwen_prediction = summarize_final_answer(qwen_summaries, question, processor, model)
            print(f"🤖 Qwen 回答：{qwen_prediction}")

            openai_summaries = run_openai_batch(image_files, question, batch_size, OPENAI_API_KEY)
            openai_prediction = summarize_final_answer(openai_summaries, question, processor, model)
            print(f"🧠 OpenAI 回答：{openai_prediction}")

            qwen_metrics = evaluate_answer(qwen_prediction, reference)
            openai_metrics = evaluate_answer(openai_prediction, reference)

            results.append({
                "id": idx,
                "question": question,
                "reference": reference,
                "qwen_prediction": qwen_prediction,
                "openai_prediction": openai_prediction,
                "evaluation": {
                    "qwen_rouge": qwen_metrics['rougeL_f1'],
                    "qwen_hit": qwen_metrics['substring_hit'],
                    "openai_rouge": openai_metrics['rougeL_f1'],
                    "openai_hit": openai_metrics['substring_hit']
                }
            })

    return results


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    csv_path = "./QA/國泰法說會簡報QA - 定位問題.csv"
    image_folder = "./output_images_fitz"

    start_time = time.time()
    results = process_csv_questions(csv_path, image_folder, processor, model, OPENAI_API_KEY, batch_size=3)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print("\n📋 每一題評估結果：")
    for r in results:
        rouge_qwen = r['evaluation']['qwen_rouge']
        hit_qwen = r['evaluation']['qwen_hit']
        rouge_openai = r['evaluation']['openai_rouge']
        hit_openai = r['evaluation']['openai_hit']
        print(f"ID {r['id']:>2} | Qwen ROUGE: {rouge_qwen:.4f}, Hit: {hit_qwen} | OpenAI ROUGE: {rouge_openai:.4f}, Hit: {hit_openai}")

    print("\n📊 Evaluation Summary:")
    total = len(results)
    avg = lambda key: round(sum(r['evaluation'][key] for r in results) / total, 4) if total else 0.0
    print(f"🔹 Qwen - Avg ROUGE: {avg('qwen_rouge')}, Avg Hit: {avg('qwen_hit')}")
    print(f"🔹 OpenAI - Avg ROUGE: {avg('openai_rouge')}, Avg Hit: {avg('openai_hit')}")
    print(f"\n⏱️ Total Runtime: {int(minutes)} min {int(seconds)} sec")
