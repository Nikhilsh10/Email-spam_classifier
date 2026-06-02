import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
import altair as alt

# Set page config for a premium portfolio experience
st.set_page_config(
    page_title="Email Spam Classifier | ML Portfolio",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/light themes and premium glassmorphism feel
st.markdown("""
<style>
    /* Main container styling */
    .reportview-container {
        background: #f8f9fa;
    }
    
    /* Styled Metric Cards */
    .metric-card {
        background-color: white;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    .metric-title {
        font-size: 14px;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .metric-spam {
        color: #dc3545;
    }
    .metric-ham {
        color: #198754;
    }
    
    /* Code/Highlight container */
    .highlight-container {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #0d6efd;
        line-height: 1.6;
        font-size: 16px;
        color: #212529;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding: 20px;
        color: #6c757d;
        font-size: 14px;
        border-top: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and vectorizer safely
@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open('spam.pkl', 'rb'))
        cv = pickle.load(open('vectorizer.pkl', 'rb'))
        return model, cv
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None

model, cv = load_assets()

# Initialize session state for classification history if not exists
if 'history' not in st.session_state:
    st.session_state.history = [
        {"email": "Congratulations! You've won a free ticket to the Bahamas. Claim your cash prize now!", "prediction": "Spam", "confidence": 0.999, "timestamp": "Recent"},
        {"email": "Hi team, just a reminder that our weekly progress meeting is scheduled for tomorrow at 10 AM.", "prediction": "Ham (Not Spam)", "confidence": 0.985, "timestamp": "Recent"}
    ]

# Sidebar navigation and info
with st.sidebar:
    st.title("✉️ Spam Guard")
    st.markdown("---")
    
    # Page selector
    app_mode = st.radio(
        "Navigation",
        ["🔍 Classify Email", "📊 Model Insights", "📁 Batch Classifier", "📜 History Logs"]
    )
    
    st.markdown("---")
    st.subheader("Model Specs")
    st.info("""
    - **Algorithm**: Multinomial Naive Bayes
    - **Feature Extractor**: Count Vectorizer (Bag of Words)
    - **Vocabulary Size**: 8,745 words
    """)
    
    st.markdown("---")
    st.markdown("Created as a portfolio project by [Nikhil Sharma](https://github.com/Nikhilsh10)")

# ----------------- PAGE 1: SINGLE EMAIL CLASSIFIER -----------------
if app_mode == "🔍 Classify Email":
    st.title("✉️ Email Spam Classifier")
    st.write("Analyze individual email messages with real-time predictions and word importance analysis (X-Ray Mode).")
    st.write("---")

    # Layout: two columns (Input, Results)
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Input Message")
        user_input = st.text_area(
            "Paste email content here:", 
            height=200, 
            placeholder="Type or paste the email content you want to inspect..."
        )
        
        classify_btn = st.button("Analyze & Classify", type="primary")
        
        # Sample Quick Test buttons
        st.markdown("**Try a sample:**")
        sample_cols = st.columns(2)
        with sample_cols[0]:
            if st.button("Sample Spam Email 🚨"):
                st.session_state.sample_text = "URGENT! Your mobile number has been selected for a £2000 bonus. Call 09061213237 now to claim your reward. T&Cs apply."
                st.rerun()
        with sample_cols[1]:
            if st.button("Sample Ham Email ✅"):
                st.session_state.sample_text = "Hey, are we still meeting up for dinner tonight? Let me know when you leave work."
                st.rerun()

        # Update input area if sample is selected
        if 'sample_text' in st.session_state:
            user_input = st.session_state.sample_text
            del st.session_state.sample_text
            st.rerun()

    with col2:
        st.subheader("Analysis & Confidence")
        if classify_btn and user_input.strip():
            # 1. Transform & Predict
            data = [user_input]
            vect = cv.transform(data).toarray()
            
            # Predict & Probabilities
            prediction = model.predict(vect)[0]
            probabilities = model.predict_proba(vect)[0]
            
            is_spam = prediction == 1
            label = "Spam" if is_spam else "Ham (Not Spam)"
            confidence = probabilities[1] if is_spam else probabilities[0]
            
            # Save to history
            st.session_state.history.insert(0, {
                "email": user_input[:100] + ("..." if len(user_input) > 100 else ""),
                "prediction": label,
                "confidence": round(float(confidence), 3),
                "timestamp": "Just Now"
            })
            
            # Display results in styled boxes
            if is_spam:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Classification</div>
                    <div class="metric-value metric-spam">🚨 SPAM</div>
                    <p style="margin: 0; color: #dc3545; font-weight: 500;">High probability of being unwanted or malicious.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Classification</div>
                    <div class="metric-value metric-ham">✅ HAM</div>
                    <p style="margin: 0; color: #198754; font-weight: 500;">Looks safe and legitimate.</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            
            # Confidence meter
            st.markdown(f"**Confidence Level**: `{confidence*100:.2f}%`")
            st.progress(float(confidence))
            
        else:
            st.info("Enter an email and click 'Analyze & Classify' to see prediction insights here.")

    # X-Ray section (only displayed after classification)
    if classify_btn and user_input.strip() and model is not None and cv is not None:
        st.write("---")
        st.subheader("🔍 Highlighted X-Ray View (Why was this classified?)")
        st.write("This mode parses each word and highlights its contribution to the classification. Words highlighted in **red** trigger the Spam filter, while words in **green** indicate legitimate (Ham) content.")
        
        # Word highlighter logic
        analyzer = cv.build_analyzer()
        tokens = set(analyzer(user_input))
        
        # Get log probs from MultinomialNB
        log_prob_ham = model.feature_log_prob_[0]
        log_prob_spam = model.feature_log_prob_[1]
        vocab = cv.vocabulary_
        
        # Find feature importance for each unique word
        word_importance = {}
        for token in tokens:
            if token in vocab:
                idx = vocab[token]
                word_importance[token] = log_prob_spam[idx] - log_prob_ham[idx]
        
        # Format the text with highlighting
        words_and_spaces = re.split(r'(\s+)', user_input)
        highlighted_html = []
        
        for item in words_and_spaces:
            if not item.strip():
                highlighted_html.append(item.replace('\n', '<br>'))
                continue
            
            # Strip punctuation for lookup
            clean_word = re.sub(r'[^\w\d]', '', item).lower()
            if clean_word in word_importance:
                imp = word_importance[clean_word]
                if imp > 0.5:  # Spam indicator
                    alpha = min(0.1 + abs(imp) * 0.12, 0.6)
                    highlighted_html.append(
                        f'<span style="background-color: rgba(220, 53, 69, {alpha:.2f}); '
                        f'padding: 2px 4px; border-radius: 4px; border-bottom: 2px solid #dc3545; '
                        f'cursor: help;" title="Spam Importance Index: {imp:.2f}">{item}</span>'
                    )
                elif imp < -0.5:  # Ham indicator
                    alpha = min(0.1 + abs(imp) * 0.12, 0.6)
                    highlighted_html.append(
                        f'<span style="background-color: rgba(25, 135, 84, {alpha:.2f}); '
                        f'padding: 2px 4px; border-radius: 4px; border-bottom: 2px solid #198754; '
                        f'cursor: help;" title="Ham Importance Index: {abs(imp):.2f}">{item}</span>'
                    )
                else:
                    highlighted_html.append(item)
            else:
                highlighted_html.append(item)
                
        output_html = "".join(highlighted_html)
        
        st.markdown(f"""
        <div class="highlight-container">
            {output_html}
        </div>
        """, unsafe_allow_html=True)
        st.caption("💡 Hover over highlighted words to see their exact importance index.")

# ----------------- PAGE 2: MODEL INSIGHTS -----------------
elif app_mode == "📊 Model Insights":
    st.title("📊 Model & Vocabulary Insights")
    st.write("Understand the backend details of the trained Multinomial Naive Bayes classifier.")
    st.write("---")
    
    if model is not None and cv is not None:
        # Vocabulary metrics
        vocab = cv.vocabulary_
        feature_names = cv.get_feature_names_out()
        log_prob_ham = model.feature_log_prob_[0]
        log_prob_spam = model.feature_log_prob_[1]
        
        # Calculate spam ratio index
        importance = log_prob_spam - log_prob_ham
        
        # Column layout
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Top Spam Indicators")
            st.write("These words have the highest relative likelihood of appearing in Spam messages.")
            
            top_spam_idx = np.argsort(importance)[::-1][:15]
            spam_df = pd.DataFrame({
                "Word": [feature_names[i] for i in top_spam_idx],
                "Log Likelihood Diff": [importance[i] for i in top_spam_idx]
            })
            
            chart_spam = alt.Chart(spam_df).mark_bar(color='#dc3545').encode(
                x=alt.X('Log Likelihood Diff:Q', title='Spam Association Strength'),
                y=alt.Y('Word:N', sort='-x', title='Word')
            ).properties(height=350)
            
            st.altair_chart(chart_spam, use_container_width=True)
            
        with c2:
            st.subheader("Top Ham Indicators")
            st.write("These words have the highest relative likelihood of appearing in legitimate (Ham) messages.")
            
            top_ham_idx = np.argsort(importance)[:15]
            ham_df = pd.DataFrame({
                "Word": [feature_names[i] for i in top_ham_idx],
                "Ham Association Strength": [abs(importance[i]) for i in top_ham_idx]
            })
            
            chart_ham = alt.Chart(ham_df).mark_bar(color='#198754').encode(
                x=alt.X('Ham Association Strength:Q', title='Ham Association Strength'),
                y=alt.Y('Word:N', sort='-x', title='Word')
            ).properties(height=350)
            
            st.altair_chart(chart_ham, use_container_width=True)
            
        st.write("---")
        st.subheader("🔍 Vocabulary Keyword Checker")
        st.write("Lookup any word to see how the model evaluates its spam-likelihood status.")
        
        search_query = st.text_input("Enter a single word to lookup:").strip().lower()
        if search_query:
            if search_query in vocab:
                idx = vocab[search_query]
                imp = importance[idx]
                
                st.write("")
                if imp > 0:
                    st.success(f"**'{search_query}'** is a **Spam indicator** (Score: `+{imp:.2f}`). It is more likely to appear in spam emails.")
                else:
                    st.info(f"**'{search_query}'** is a **Ham indicator** (Score: `{-imp:.2f}`). It is more likely to appear in genuine emails.")
            else:
                st.warning(f"The word '{search_query}' was not found in the model's training vocabulary.")
    else:
        st.error("Assets not loaded.")

# ----------------- PAGE 3: BATCH CLASSIFIER -----------------
elif app_mode == "📁 Batch Classifier":
    st.title("📁 Batch Text Classifier")
    st.write("Process multiple emails at once by pasting them or uploading a text file.")
    st.write("---")
    
    st.subheader("Input Emails")
    st.write("Enter multiple emails. Separate each email with a line containing `---` (three hyphens).")
    
    batch_input = st.text_area(
        "Paste batch emails here:",
        height=250,
        placeholder="Email 1 text here...\n---\nEmail 2 text here...\n---\nEmail 3 text here...",
        value="URGENT! You have won a 1-week family cruise holiday to Hawaii. Text CRUISE to 80099 to claim your tickets.\n---\nDear Nikhil, are we on track for the repository cleanup this evening? Please send updates.\n---\nGet cheap luxury replica watches today! Rolex, Omega, Cartier starting at $99. Fast shipping."
    )
    
    if st.button("Run Batch Classification", type="primary") and batch_input.strip():
        emails = [email.strip() for email in batch_input.split("---") if email.strip()]
        
        results = []
        for i, email in enumerate(emails):
            vect = cv.transform([email]).toarray()
            pred = model.predict(vect)[0]
            probs = model.predict_proba(vect)[0]
            
            is_spam = pred == 1
            confidence = probs[1] if is_spam else probs[0]
            
            results.append({
                "Index": i + 1,
                "Email Snippet": email[:70] + ("..." if len(email) > 70 else ""),
                "Classification": "🚨 Spam" if is_spam else "✅ Ham",
                "Confidence": f"{confidence*100:.1f}%"
            })
            
        df_results = pd.DataFrame(results)
        
        # Display summary metrics
        total = len(results)
        spam_count = sum(1 for r in results if "Spam" in r["Classification"])
        ham_count = total - spam_count
        
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Total Processed", total)
        m_col2.metric("Spam Found", f"{spam_count} ({spam_count/total*100:.1f}%)")
        m_col3.metric("Legitimate (Ham)", f"{ham_count} ({ham_count/total*100:.1f}%)")
        
        st.write("")
        st.subheader("Classification Table")
        st.dataframe(df_results, use_container_width=True)
        
        # Download Results
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="spam_classification_results.csv",
            mime="text/csv"
        )

# ----------------- PAGE 4: HISTORY LOGS -----------------
elif app_mode == "📜 History Logs":
    st.title("📜 Classification History")
    st.write("Track the history of emails classified during this session.")
    st.write("---")
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No classification history logged in this session yet.")

# Footer markup for portfolio layout
st.markdown("""
<div class="footer">
    <p>Email Spam Classifier Portfolio Project. Model trained using Scikit-Learn. Frontend powered by Streamlit.</p>
</div>
""", unsafe_allow_html=True)
