

- 用 WSL2 切到 backend 資料夾，再下指令： `python3 -m venv venv`
- 透過這個指令： `source venv/bin/activate` 進到 venv (似乎也有啟用 venv 的感覺，之後再釐清，先讓專案跑起來)
- `pip install fastapi uvicorn python-dotenv` 安裝套件
- 用指令： `pip freeze > requirements.txt` 產生 `requirements.txt`



## 如何將後端 api 服務在本地端運行起來？
1. 在 `ask-your-file` 路徑底下運行指令： `source backend/venv/bin/activate`
   - windows 建議在 WSL2 中執行
2. 運行指令：`python -m backend`  
  即可將後端的服務跑起來

