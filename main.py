import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline


class Item(BaseModel):
   text: str

app = FastAPI()
generator = pipeline('text-generation', model='gpt2')

@app.get("/")
def root():
   return {"message": "English text generator by example"}

@app.post("/generate/")
async def predict(item: Item, length: int = 120, num_seq: int = 10):
   output = generator(item.text, max_length=length, num_return_sequences=num_seq)
   text = {}
   for text_part in range(len(output)):
      text[f"text_part_{text_part}"] = output[text_part].get('generated_text') + '\n'
   return {"generation_parameters": {"length": length, "num_seq": num_seq}, "generated_text": text}
