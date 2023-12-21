from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import requests
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
from utils_vec import cosine_similarity, euclidean, get_vectors



app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

dataset = pd.read_csv('capitals.txt', delimiter=' ')
dataset.columns = ['city1', 'country1', 'city2', 'country2']

word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb"))

from fastapi import HTTPException

@app.post("/getCountry")
async def getCountry(city1: str, country1: str, city2: str):
    try:
        group = {city1, country1, city2}

        with open("word_embeddings_subset.p", "rb") as file:
            embeddings = pickle.load(file)

        city1_emb = word_embeddings.get(city1)
        country1_emb = word_embeddings.get(country1)
        city2_emb = word_embeddings.get(city2)

        if any(emb is None for emb in [city1_emb, country1_emb, city2_emb]):
            raise HTTPException(status_code=400, detail="Invalid city or country names")

        vec = country1_emb - city1_emb + city2_emb
        vec = vec.astype(float)  # Convertir a float

        similarity = -1
        best_match = None

        for word, word_emb in embeddings.items():
            if word not in group:
                cur_similarity = cosine_similarity(vec, word_emb)

                if cur_similarity > similarity:
                    similarity = cur_similarity
                    best_match = (word, similarity)

        return best_match

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run(app, host='localhost' ,port=8000)