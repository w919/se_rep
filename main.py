import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, set_seed


class Item(BaseModel):
   text: str

app = FastAPI()
generator = pipeline('text-generation', model='gpt2')

@app.get("/")
def root():
   return {"message": "English text generator by example"}

@app.post("/generate/")
async def predict(item: Item):
   length = 120
   num_seq = 10
   output = generator(item.text, max_length=length, num_return_sequences=num_seq)
   text = ''
   for i in range(len(output)):
      text += (output[i].get('generated_text') + '\n')
   return {"generated_text": text}
