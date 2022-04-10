import bentoml
from transformers import (BartForConditionalGeneration, PreTrainedTokenizerFast)
import os


def load_model():
    model = BartForConditionalGeneration.from_pretrained('./model')
    return model

model = load_model()
tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join('/content/emji_tokenizer', 'model.json'),
            bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
tag = bentoml.transformers.save('chat_bart', model = model, tokenizer = tokenizer)
