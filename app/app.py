import streamlit as st
import torch
from model import predict_sentiment
from model import get_model
from model import get_tokenizer

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/Kasa-Harendra/IMDB-sentiment-analysis',
        'About': "# Sentiment Analysis App\nBuilt with PyTorch and Streamlit"
    }
)

st.title("Sentiment Analysis")

st.header("Model Prediction")
sentiment = st.text_input("Enter the text...")

if 'model' not in st.session_state:
    st.session_state['model'] = get_model()

if 'tokenizer' not in st.session_state
    st.session_state['tokenizer'] = get_tokenizer()

if st.button(label="Classify"):
    if sentiment.strip():
        with st.spinner("Analyzing sentiment..."):
            try:
                result = predict_sentiment(sentiment)
                
                sentiment_label = result["sentiment"]
                confidence_score = result["confidence"]
                
                if sentiment_label == "Positive":
                    st.success(f"**Prediction:** {sentiment_label}")
                else:
                    st.error(f"**Prediction:** {sentiment_label}")
            
                st.info(f"**Input Text:** {sentiment}")
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please enter some text to analyze.")

st.divider()

st.header("Code")
st.code(
    body="""
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
            pooled_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0][:, 0]
            output = self.model_extrension(pooled_out)
            return output
    """,
    language="python"
)

st.divider()

st.header("Google-BERT Model Info")
with st.container():
    st.markdown(
        """
        `google-bert/bert-base-uncased` is one of the original and most widely used versions of Google's BERT (Bidirectional Encoder Representations from Transformers) model. It's a foundational model in Natural Language Processing (NLP).

- `bert`: The core architecture. Its key innovation was its bidirectional nature, meaning it understands the context of a word by considering both the words that come before and after it. This provides a much deeper understanding of language compared to previous models that only looked in one direction.
- `base`: Refers to the model's size. The "base" version has 12 transformer layers and about 110 million parameters. This makes it smaller and faster than the bert-large version, offering a good balance between performance and computational cost.
- `uncased`: Indicates that all the text used to train the model was converted to lowercase. As a result, the model does not distinguish between "Apple" and "apple". This simplifies the vocabulary but makes it unsuitable for tasks where capitalization is important (e.g., identifying proper nouns).
        """)
    
    st.markdown("""
## Sentiment Analysis Model Architecture

The model consists of two primary parts:  
- **BERT Backbone** for feature extraction  
- **Classification Head** for prediction  

### a. BERT Backbone (`self.bert`)

- **What it is**:  
  The foundation of our model is **`google-bert/bert-base-uncased`**, a powerful pre-trained language model from Google.

- **Its Role**:  
  It acts as a sophisticated **feature extractor**. When we feed it a text sequence (represented by `input_ids` and an `attention_mask`), its job is to convert that text into **rich, contextualized embeddings**.  
  It understands grammar, context, and subtle word meanings.

### b. Classification Head (`self.model_extrension`)

- **What it is**:  
  A small, custom-built neural network defined using `nn.Sequential`, placed on top of the BERT backbone.

- **Its Role**:  
  It takes the features extracted by BERT and performs the final **sentiment classification**.

#### 🔎 Layer-by-Layer Breakdown:

1. **`nn.Linear(768, 384)`**  
   - Fully connected linear layer.  
   - Takes the 768-dimensional [CLS] output from BERT and projects it into a 384-dimensional vector.  
   - The **768 size** is specific to the `bert-base` architecture.

2. **`nn.Dropout(0.25)`**  
   - Regularization technique to prevent overfitting.  
   - Randomly sets **25% of neurons** to zero during training.  
   - Helps the model learn **robust, generalizable features**.

3. **`nn.Linear(384, 1)`**  
   - Fully connected layer.  
   - Reduces the 384-dimensional vector down to **1 value (a logit)**, representing sentiment.

4. **`nn.Sigmoid()`**  
   - Final activation function.  
   - Squashes the output between **0 and 1**, perfect for binary classification.  
   - **Interpretation**:  
     - Value close to **0 → Negative sentiment**  
     - Value close to **1 → Positive sentiment**

### Forward Pass 

The `forward` method defines how data flows through the model during prediction.

1. **Input**  
   - `input_ids`: Numerical representation of text tokens.  
   - `attention_mask`: Indicates which tokens are real words vs. padding.  

2. **BERT Processing**  
   - Inputs are passed to `self.bert`.  
   - `...[0][:, 0]` explained:  
     - `[0]` → Selects hidden states of all tokens.  
     - `[:, 0]` → Extracts the embedding of the **[CLS] token**.  
     - The **[CLS] token** serves as a sentence-level representation for classification.

3. **Classification**  
   - The `[CLS]` embedding (size `768`) is passed into the custom `self.model_extrension`.

4. **Final Output**  
   - Model returns a **floating-point value between 0 and 1**, representing the predicted sentiment score.

---

### 🔄 Visual Flow

``` Input Text → Tokenizer → [input_ids, attention_mask] → BERT Model → [CLS] Token → Classification Head → Sentiment Score (0–1) ```
        
        """
    )
    
st.header("Model Structure: ")
st.code(
    """
SentimentModel(
    (bert): BertModel(
        (embeddings): BertEmbeddings(
            (word_embeddings): Embedding(30522, 768, padding_idx=0)
            (position_embeddings): Embedding(512, 768)
            (token_type_embeddings): Embedding(2, 768)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): BertEncoder(
            (layer): ModuleList(
                (0-11): 12 x BertLayer(
                    (attention): BertAttention(
                        (self): BertSdpaSelfAttention(
                            (query): Linear(in_features=768, out_features=768, bias=True)
                            (key): Linear(in_features=768, out_features=768, bias=True)
                            (value): Linear(in_features=768, out_features=768, bias=True)
                            (dropout): Dropout(p=0.1, inplace=False)
                        )
                        (output): BertSelfOutput(
                            (dense): Linear(in_features=768, out_features=768, bias=True)
                            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                            (dropout): Dropout(p=0.1, inplace=False)
                        )
                    )
                    (intermediate): BertIntermediate(
                        (dense): Linear(in_features=768, out_features=3072, bias=True)
                        (intermediate_act_fn): GELUActivation()
                    )
                    (output): BertOutput(
                        (dense): Linear(in_features=3072, out_features=768, bias=True)
                        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                        (dropout): Dropout(p=0.1, inplace=False)
                    )
                )
            )
        )
        (pooler): BertPooler(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (activation): Tanh()
            )
        )
    (model_extension): Sequential(
        (0): Linear(in_features=768, out_features=384, bias=True)
        (1): Dropout(p=0.25, inplace=False)
        (2): Linear(in_features=384, out_features=1, bias=True)
        (3): Sigmoid()
    )
)
    """
)