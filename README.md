# India Stance Classifier (Streamlit + SVM)

A simple Streamlit web app that classifies a post/tweet as **Pro-India**, **Anti-India**, or **Neutral**.

## Files Needed
- `app.py` (Streamlit app)
- `requirements.txt` (dependencies)
- `pksvm_model.pkl` (your SVM classifier)
- `vectorizer.pkl` (the TF-IDF vectorizer you used during training)

## Run Locally
```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
