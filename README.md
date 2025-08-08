# chart-vlm
vlm on chart comprehension

# 這個檔案只會最簡單的說明下列兩點，剩下的細節，會放在各自的資料夾中的 README
1. 後端服務怎麼跑起來？
2. 前端服務怎麼跑起來？
3. 前端目前設定在哪個 port；如果要調整的話，該如何調整？

## 後端服務怎麼跑起來
1. 在 `ask-your-file` 路徑底下運行指令： `source backend/venv/bin/activate`
   - windows 建議在 WSL2 中執行
2. 運行指令：`python -m backend`  
  即可將後端的服務跑起來

## 前端服務怎麼跑起來
1. 在 `ask-your-file` 路徑底下運行指令： `cd frontend` 進到 frontend 資料夾
2. 運行指令：`yarn dev`  
  即可將前端的服務跑起來

## 前端
### 目前設定在哪個 port？
- localhost 5555

### 該如何調整前端的 port？
調整 `frontend/vite.config.js` 內的 `port` 屬性值  
(應該會在第 8 行)
