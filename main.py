import json
import random
import string

import numpy as np
import requests
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
   """Generate text from given string

   Args:
       item (Item): object with input string
       length (int, optional): Lenght of text generation part. Defaults to 120.
       num_seq (int, optional): Numbers of text generations parts. Defaults to 10.

   Returns:
       dict: json-like answer to client
   """
   output = generator(item.text, max_length=length, num_return_sequences=num_seq)
   text = {}
   for text_part in range(len(output)):
      text[f"text_part_{text_part}"] = output[text_part].get('generated_text') + '\n'
   return {"generation_parameters": {"length": length, "num_seq": num_seq}, "generated_text": text}

@app.get("/random_generate/")
async def random_generate(length: int = 120, num_seq: int = 10) -> dict:
   """Get generation from wikipedia summary of random article

   Args:
       length (int, optional): Lenght of text generation part. Defaults to 120.
       num_seq (int, optional): Numbers of text generations parts. Defaults to 10.

   Returns:
       dict: json-like answer to client
   """
   wiki_url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
   wiki_answer = requests.get(wiki_url)
   inp_text = wiki_answer.json()["extract"]
   output = generator(inp_text, max_length=length, num_return_sequences=num_seq)
   text = {}
   for text_part in range(len(output)):
      text[f"text_part_{text_part}"] = output[text_part].get('generated_text') + '\n'
   return {
      "generation_parameters": {"length": length, "num_seq": num_seq}, 
      "input_text": inp_text,
      "generated_text": text
   }
