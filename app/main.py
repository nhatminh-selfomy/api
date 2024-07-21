from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
import sys
import os

# Adding the path to the models directory
sys.path.append(os.path.abspath('../src/'))
# from models import GOPT
from app.startup import load_model
from app.utils import audio_to_tensor

app = FastAPI()

# CORS setup (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this according to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
gopt = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global gopt
    gopt = load_model()
    yield
    # Clean up any resources if needed

app.router.lifespan_context = lifespan

@app.get("/")
async def read_root():
    return {"message": "Welcome to the GOPT model API"}

@app.post("/score_pronunciation")
async def score_pronunciation(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_location = f"audio/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        
        # Process the audio file
        input_tensor = audio_to_tensor(file_location)
        
        # Run the model
        with torch.no_grad():
            gopt.eval()
            output = gopt(input_tensor)
        
        # Assuming the model's output contains a pronunciation score
        score = output.item()  # Adjust this according to your model's output structure
        
        return {"pronunciation_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
