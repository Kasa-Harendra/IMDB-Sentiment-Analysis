import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch
import streamlit as st
import os
import gdown

if not os.path.exists('model/sentiment_model.pth'):
    os.makedirs(name='model', exist_ok=True)
    
    file_id = '1BnHzaWlTFaAic3gnm_1ALKzlZti642Wo'  
    url = f'https://drive.google.com/uc?id={file_id}'
    
    st.toast('Downloading model')
    
    gdown.download(url, 'model/sentiment_model.pth', quiet=False)

if 'tokenizer' not in st.session_state:
    st.session_state['tokenizer'] = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="google-bert/bert-base-uncased")
bert_model = AutoModel.from_pretrained(pretrained_model_name_or_path="google-bert/bert-base-uncased")

class SentimentModel(nn.Module):
    def __init__(self, bert):
        super(SentimentModel, self).__init__()
        self.bert = bert
        self.model_extrension = nn.Sequential(
            nn.Linear(768, 384),
            nn.Dropout(0.25),
            nn.Linear(384, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        outs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0][:, 0]
        output = self.model_extrension(outs)
        return output

    
model = SentimentModel(bert_model)

model.load_state_dict(torch.load("model/sentiment_model.pth", map_location=torch.device('cpu')))
st.toast('Model Loaded Succesfully')

if 'model' not in st.session_state:
    st.session_state['model'] = model 

def predict_sentiment(text):
    st.session_state['model'].eval()
    with torch.no_grad():
        encoded = st.session_state['tokenizer'](
            text,
            padding='max_length',
            truncation=True,
            max_length=200,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        output = model(input_ids, attention_mask)
        confidence = output.item()
        pred = (output > 0.5).float().item()
    
    sentiment = "Positive" if pred == 1 else "Negative"
    return {"sentiment": sentiment, "confidence": confidence}


