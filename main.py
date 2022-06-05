from fastapi import FastAPI
from pydantic import BaseModel

import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

import os

app = FastAPI()

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./model')
    return model


model = load_model()
tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join('./tokenizer', 'model.json'),
            bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')


class Chat(BaseModel):
    sent: str


@app.post("/predict/")
async def create_context(chat : Chat):

    context = {}
    
    input_ids =  [tokenizer.bos_token_id] + tokenizer.encode(chat.sent) + [tokenizer.eos_token_id]
    res_ids = model.generate(torch.tensor([input_ids]),
                            max_length=128,
                            num_beams=5,
                            eos_token_id=tokenizer.eos_token_id,
                            bad_words_ids=[[tokenizer.unk_token_id]])
    result = tokenizer.batch_decode(res_ids.tolist())[0]
    result = result[6:-4]
    context['response'] = result
        
    return context
