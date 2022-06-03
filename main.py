import os

import gluonnlp as nlp
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from fastapi import FastAPI
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from pydantic import BaseModel
from torch import cuda, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast


# Chatbot Model load
app = FastAPI()
model = load_model()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join("./tokenizer", "model.json"),
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)


# Bert Model load
bertmodel, vocab = get_pytorch_kobert_model()
device = 'cuda:0' if cuda.is_available() else 'cpu'
bertModel = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
bertModel.load_state_dict(
    torch.load("./kobert/sentiment_classifier_version_1.pt")
)
#Tokenizer
bert_tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(bert_tokenizer, vocab, lower=False)


class Chat(BaseModel):
    sent: str


class ChatList(BaseModel):
    sent: list


@app.post("/predict/")
async def create_context(chat: Chat):

    context = {}
    input_ids = (
        [tokenizer.bos_token_id]
        + tokenizer.encode(chat.sent)
        + [tokenizer.eos_token_id]
    )
    res_ids = model.generate(
        torch.tensor([input_ids]),
        max_length=128,
        num_beams=5,
        eos_token_id=tokenizer.eos_token_id,
        bad_words_ids=[[tokenizer.unk_token_id]],
    )
    result = tokenizer.batch_decode(res_ids.tolist())[0]
    result = result[6:-4]
    context["response"] = result

    return context


@app.post("/sentiment/")
async def create_context(chat: ChatList):

    context = {}
    data = [[sent, 0] for sent in ChatList]

    data_test =  BERTDataset(data, 0, 1, tok, 128, True, False)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=32, num_workers=5)

    model.eval()
    list_pred = []
    model_pred(list_pred, test_dataloader)

    context["response"] = 0 if list_pred[0][0] > list_pred[0][1] else 1

    return context

def load_model():
    model = BartForConditionalGeneration.from_pretrained("./model")
    return model


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


def model_pred(arg_list, dataloader):
  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(dataloader):
    token_ids = token_ids.long().to(device)
    valid_length = valid_length
    segment_ids = segment_ids.long().to(device)
    label = label.long().to(device)
    pred_var = model(token_ids, valid_length, segment_ids)
    print(pred_var)
    _, predict = torch.max(pred_var,1)
    arg_list.extend(pred_var.tolist())