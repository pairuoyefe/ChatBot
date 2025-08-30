# Star Wars Chatbot – Intent Classification  

## Project Overview  

This project explores building a chatbot for Star Wars conversations using two different approaches:  

- Deep Learning (TensorFlow / Keras Neural Network)  
- Classical Machine Learning (Random Forest Classifier with TF-IDF)  

The goal was to predict user intents (e.g., greetings, Star Wars references, etc.) and provide an appropriate response.  

---

## Methodology  

### 1. Data  
- JSON file: `starwarsintents.json`  
- Contains:  
  - **Patterns** → Example user queries  
  - **Tag** → Intent label  
  - **Responses** → Chatbot replies  

### 2. Preprocessing  
- Text tokenization & padding (for NN).  
- TF-IDF Vectorization (for ML models).  
- Label encoding with `sklearn`.  

### 3. Models  

**Neural Network (TensorFlow/Keras)**  
- Dense(64, relu) → Dense(output, softmax)  
- Optimizer: Adam  
- Loss: Sparse categorical crossentropy  

**Random Forest Classifier**  
- TF-IDF features  
- Train/test split: 80/20  

---

## Results  

### TensorFlow Neural Network  
- Train Accuracy: ~0.60  
- Validation Accuracy: ~0.53  

### Random Forest Classifier  
- Accuracy: ~0.79  

| Model            | Accuracy | Notes                                    |
|------------------|----------|------------------------------------------|
| TensorFlow NN    | ~0.60    | Underperformed due to limited data.      |
| Random Forest    | ~0.79    | Stronger performance on small dataset.   |
<img width="488" height="357" alt="RandomF" src="https://github.com/user-attachments/assets/53c6d416-7b93-4c4a-9b65-a8af9a0c2c0a" />
<img width="627" height="363" alt="tensoflow" src="https://github.com/user-attachments/assets/143e8a7e-3b8e-436a-b909-c769d7bbe590" />


---

## Challenges  

- **Limited dataset** → The small number of intents/patterns reduced model generalization.  
- Neural network struggled to capture intent meaning with so little data.  

---

## Recommendations & Future Work  

1. **Expand dataset**  
   - Add more intents, patterns, and responses.  
   - Scrape or crowdsource Star Wars dialogues for training.  

2. **Try advanced architectures**  
   - LSTM / GRU for sequential context.  
   - Transformers (BERT / DistilBERT) for better text understanding.  

3. **Hybrid approach**  
   - Use Random Forest for baseline + Neural Nets for fine-grained intents.  

4. **Deployment**  
   - Wrap model into a Flask / FastAPI API.  
   - Deploy on Streamlit or as a Telegram/Discord bot.  

---

## Saved Models  

- `rf_chatbot_model.pkl` → Random Forest model.  
- `tfidf_vectorizer.pkl` → TF-IDF vectorizer.  
- `label_encoder.pkl` → Encoded labels.  

---

## Example Interaction  

