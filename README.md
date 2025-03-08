# 📩 Spam Detection using NLP & Machine Learning
___
## Introduction
Spam messages have become a major problem in today's digital world, often leading to scams, phishing attacks, and unnecessary clutter in our inboxes. To tackle this issue, I built a **machine learning-based spam detection model** that can classify **emails and SMS messages as spam or not** using **Natural Language Processing (NLP)**.

For this project, I used **TF-IDF vectorization** to extract important features from the text and trained a **Multinomial Naive Bayes (MultinomialNB) classifier**, which is well-suited for text classification tasks. After testing different techniques, I found that **TF-IDF significantly improved performance** compared to **CountVectorizer**.

Since marking a **legitimate message as spam (false positive) can have serious consequences***, I focused on **optimizing the precision score** to reduce such errors and ensure more accurate filtering.
___
## Key Features
* **Dataset:** SMS/Email spam dataset sourced from Kaggle [here](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)
* **Feature Engineering:** I have exracted lexical features from the text data.TF-IDF Vectorizer for text transformation
* **Machine Learning Model:** Multinomial Naive Bayes (with other models explored)
* **Evaluation Metric:** Precision-focused approach to reduce false positives
___
## Model Performance
You can view below images that I directly copy from the notebook:

![Image of metrics calculation](https://github.com/bijaypokhrel05/Email-SMS-Spam-Detection/blob/main/dataframe_of_performance_metrics.png)
___
## How to Run the Project
1. **Clone the repository:**
   ```bash
   git clone https://github.com/bijaypokhrel05/Email-SMS-Spam-Detection.git
   cd spam-detection-nlp
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run jupyter notebook for training/testing:**
   ```bash
   jupyter notebook
   ```
4. **Run the app for real-time prediction:**
   ```bash
   streamlit run app.py
   ```
___
### Conclusion
This project demonstrates how **Natural Language Processing (NLP) and Machine Learning** can effectively classify spam messages. **MultinomialNB with TF-IDF** proves to be a strong combination for this task. Future improvements could include **deep learning-based approaches like LSTMs or transformers** for more advanced detection.
