from openai import OpenAI
import re
import os
import pandas as pd

# Evaluation prompt

eval_prompt = """
你需要比較「正確答案」與 LLM 回覆，並評估回覆中是否存在「幻覺」情況。
幻覺定義是指 LLM 的回覆中包含無法從正確答案合理推論或不一致的資訊。

輸出格式：
評估分數: <0-5 數字>
原因：<簡要說明>

評估標準：
0：回覆完全與正確答案一致，無需推論
1：主要一致，少量補充或推論
2：大部分一致，仍有部分資訊不在正確答案中
3：部分一致，需要較多推理或補充
4：僅少量一致，大部分為推測或補充
5：完全不一致或回覆無法從正確答案推論
"""

model = "gpt-4o-mini" #TODO: 指定模型名稱
answer_col = "Answer" #TODO: 正確答案欄位
llm_col = "LLM_Response" #TODO: LLM 回覆欄位

# Initialization
def get_gpt_response(prompt: str, model: str = model) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

# Batch evaluation function
def llm_evaluation(df, model: str = model):
    evaluations, scores = [], []
    total = len(df)
    for i in range(total):
        correct = df.at[i, answer_col]
        reply   = df.at[i, llm_col]
        prompt  = (
            eval_prompt
            + f"\n# ground_truth:\n{correct}"
            + f"\n# llm_response:\n{reply}"
        )
        eval_text = get_gpt_response(prompt, model)
        m = re.search(r"評估分數:\s*([0-5](?:\.[0-9]+)?)", eval_text)
        score  = float(m.group(1)) if m else None
        evaluations.append(eval_text)
        scores.append(score)
    return evaluations, scores

# Notebook 執行範例：
# 載入資料並執行評估，會自動輸出 CSV 與 Excel

df = pd.read_csv("input.csv")  #TODO: 指定檔案路徑 確保包含 answer_col 與 llm_col 欄位。

evals, score_list = llm_evaluation(df)
print("Scores:", score_list)

# 將結果附加至 DataFrame
df['Evaluation'] = evals
df['Score'] = score_list

# 輸出到 CSV
output_csv = 'evaluation_results.csv' #TODO: 指定輸出檔案路徑
df.to_csv(output_csv, index=False)
# 同時輸出到 Excel
output_excel = 'evaluation_results.xlsx' #TODO: 指定輸出檔案路徑
df.to_excel(output_excel, index=False)

print(f"評估完成，CSV 儲存至 {output_csv}，Excel 儲存至 {output_excel}")