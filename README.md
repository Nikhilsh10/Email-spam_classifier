# ✉️ Email Spam Classifier Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40.1-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) 

An interactive Machine Learning dashboard to classify emails as **Spam** or **Ham (Legitimate)**, complete with word-level explanation capabilities (X-Ray Mode) and a model insights explorer.

---

## 🌟 Key Features

- **🔍 Single Email Classifier**: Paste any email to get instantly predicted probability scores and visual indicators.
- **👁️ X-Ray Explanation Mode**: Highlights specific keywords in **red** (Spam indicators) and **green** (Ham indicators) based on Naive Bayes feature probabilities. Hover to see the exact impact index of each word.
- **📊 Interactive Model Insights**: Visualizes the top 15 words that contribute most heavily to classification decisions using Altair charts. Include keyword lookup to query the model's vocabulary.
- **📁 Batch Classifier**: Upload or paste multiple emails simultaneously, visualize overall distribution percentages, and export results as a CSV.
- **📜 Session Logs**: Keeps track of recent predictions during the browser session for easy review.

---

## 🤖 The Machine Learning Model

The backend leverages a classic text classification pipeline:
1. **Feature Extraction**: `CountVectorizer` (Bag of Words) maps raw text into a vocabulary of **8,745** distinct tokens.
2. **Classifier**: `Multinomial Naive Bayes (MultinomialNB)`. Naive Bayes is exceptionally well-suited for spam filters due to its fast training speeds and solid performance under sparse frequency counts.
3. **Interpretability**: By calculating $\log P(\text{word} | \text{Spam}) - \log P(\text{word} | \text{Ham})$, we can isolate the exact contribution of each keyword, which powers the highlighting tool.

---

## 📂 Project Structure

```text
├── .gitignore            # Ignored caches and local settings
├── app.py                # Premium Streamlit Dashboard application
├── requirements.txt     # Project dependencies list
├── spam.pkl              # Pre-trained Multinomial Naive Bayes model
├── vectorizer.pkl        # Fitted CountVectorizer vocabulary
└── README.md             # Project documentation (this file)
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Nikhilsh10/Email-spam_classifier.git
cd Email-spam_classifier
```

### 2. Create and activate a Virtual Environment (Optional but recommended)
On Windows:
```bash
python -m venv myenv
.\myenv\Scripts\activate
```
On macOS/Linux:
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: If `requirements.txt` is not found, you can run `pip install streamlit scikit-learn numpy pandas altair`)*

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Nikhilsh10/Email-spam_classifier/issues).

## 📄 License
This project is [MIT](https://opensource.org/licenses/MIT) licensed.
