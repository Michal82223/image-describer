from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # zmień na swój frontend w razie potrzeby
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Opisz dokładnie co znajduje się na tym zdjęciu. Bądź szczegółowy.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_image}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 500,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
    data = response.json()
    return {"description": data["choices"][0]["message"]["content"]}
