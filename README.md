# Spam_Prediction
# 📧 SMS Spam Detection using NLP & Machine Learning

This project classifies SMS messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) and Machine Learning.

---

## 🎯 Objective

To build a binary classification model that can detect whether an incoming SMS is **spam** or **legitimate**, based on its text content.

---

## 📂 Dataset

- **Source**: [Kaggle – SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Rows:** ~5,500 messages  
- **Features:** One column of text (message), one target column (label: `spam` or `ham`)

---

## 🛠️ Tools & Libraries Used

- Python  
- pandas, numpy  
- scikit-learn (CountVectorizer, MultinomialNB, train_test_split, accuracy_score)  
- nltk (stopwords, stemming, tokenization)

---

## ⚙️ Workflow Steps

1. **Data Loading**
   - Loaded dataset into a DataFrame
   - Renamed columns for clarity

2. **Text Preprocessing (NLP)**
   - Lowercasing
   - Removing punctuation & special characters
   - Removing stopwords
   - Tokenization
   - Stemming

3. **Feature Extraction**
   - Used `CountVectorizer` to convert text into numerical features

4. **Model Training**
   - Applied `Multinomial Naive Bayes` (best for text classification)
   - Train-test split: 80% training, 20% testing

5. **Model Evaluation**
   - Evaluated using `accuracy_score`
   - Optional: Confusion matrix, classification report

---

## ✅ Results

- **Model Used**: Multinomial Naive Bayes  
- **Test Accuracy**: 🌟 **XX.XX%** *(replace with your actual value)*

---

## 📁 Files Included

- `Project-3_Spam_Prediction.ipynb` – Main Jupyter Notebook    
- `README.md` – Project description

---

## 🧠 What I Learned

- Preprocessing raw text using NLP techniques  
- Converting text into vectors using CountVectorizer  
- Building and evaluating a spam classifier  
- Importance of cleaning textual data in ML projects

---

## 🔮 Future Enhancements

- Use `TfidfVectorizer` for better feature extraction  
- Try other models like SVM, Random Forest  
- Deploy as a **Streamlit app** to test in real-time  
- Add confusion matrix, F1 score, ROC curve

---

## 🤝 Connect with Me

- GitHub: [github.com/yourusername](https://github.com/Athar-cell)  


---

> 🚀 Made with ❤️ using Python, NLP, and ML
