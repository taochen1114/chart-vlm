import time
from PIL import Image
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

# 測試圖片路徑
image_path = "./data/pdf_img/1Q25_MP_Chinese_Vupload/page_1.png"
img = Image.open(image_path).resize((720, 720))

# 問題
question = "這一頁是否與『每股盈餘』有關？請只回答是或否"

# ===== 初始化模型 =====
model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-V-2_6',
    trust_remote_code=True,
    attn_implementation='sdpa',
    torch_dtype=torch.bfloat16
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

# 包裝成 VLM 推論函式（用 MiniCPM）
def vlm_answer_debug(image, question):
    full_prompt = f"{question}\n\n請只回答「是」或「否」。不要加其他文字。"
    msgs = [{'role': 'user', 'content': [image, full_prompt]}]

    print("🚀 發送 VLM...")
    start = time.time()
    with torch.no_grad():
        answer = model.chat(
            image=image,  # 重點：這裡一定要有 image
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.2,
            max_new_tokens=128
        )
    end = time.time()
    print(f"✅ 回答完成，耗時 {end - start:.2f} 秒")
    print("📣 模型回答：", answer)
    return answer

# 執行測試
vlm_answer_debug(img, question)
