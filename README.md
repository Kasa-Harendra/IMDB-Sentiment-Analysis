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
