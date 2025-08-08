import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS").split(",")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_api():
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000)


@app.get("/")
def root():
    return {"msg": "OK"}


@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...), question: str = Form(...)):
    filename = file.filename
    content_type = file.content_type

    mock_response = {
        "filename": filename,
        "content_type": content_type,
        "question": question,
        "answer": "你問了一個很棒的問題",
        "answer_page": 10,
    }

    return mock_response
