from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

app = FastAPI()

# Obsługa CORS dla Fluttera
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ładowanie modelu (BLIP do opisu obrazów)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.post("/describe")
async def describe_image(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)

    return {"description": description}
