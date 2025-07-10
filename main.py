from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from PIL import Image
import io
import gc

app = FastAPI()

# 🔒 Zezwól na połączenia z aplikacji mobilnej (wszystkie źródła w dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lub ogranicz do domeny frontu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Lekki model do opisu obrazów
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

@app.post("/describe")
async def describe_image(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))  # 🔧 zmniejszamy obraz

        result = captioner(image)
        gc.collect()  # 🧹 czyścimy pamięć

        return {"description": result[0]["generated_text"]}

    except Exception as e:
        return {"error": str(e)}
