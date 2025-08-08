from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import os

# 載入模型與處理器
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# 所有 6 個資料夾路徑
image_folders = [
    "./output_images/1Q25_MP_Chinese_Vuploadbatch_1_images/",
    "./output_images/1Q25_MP_Chinese_Vuploadbatch_2_images/",
    "./output_images/1Q25_MP_Chinese_Vuploadbatch_3_images/",
    "./output_images/1Q25_MP_Chinese_Vuploadbatch_4_images/",
    "./output_images/1Q25_MP_Chinese_Vuploadbatch_5_images/",
    "./output_images/1Q25_MP_Chinese_Vuploadbatch_6_images/",
]

# 最終要問的問題
final_question = "文件中「每股盈餘」的數據出現在簡報的哪一頁？"

# === 分批建立對話 ===
dialog = []

for i, folder in enumerate(image_folders):
    image_files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])[:10]  # 每批限制圖片數（可試 2~4）

    # 轉為 PIL 並 resize
    images = [Image.open(p).convert("RGB").resize((720, 720)) for p in image_files]

    # 加入一輪訊息（模擬讓 VLM 記住這批圖）
    message_content = [{"type": "image", "image": img} for img in images]
    message_content.append({"type": "text", "text": f"這是第 {i+1} 批圖表資料，請記住它們。"})
    
    dialog.append({
        "role": "user",
        "content": message_content
    })

# 最後加入真正的問題
dialog.append({
    "role": "user",
    "content": [{"type": "text", "text": final_question}]
})

try:
    # 編碼處理
    text = processor.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(dialog)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    # 推理
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=400)

    # 解碼
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 輸出
    print("\n📊 問題：", final_question)
    print("🧠 回答：", output_text.strip())
    print("=" * 100)

except torch.cuda.OutOfMemoryError:
    print("❌ CUDA 記憶體不足，請降低每批圖片數或縮圖尺寸")
    torch.cuda.empty_cache()
