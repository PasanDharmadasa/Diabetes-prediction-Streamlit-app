# Diabetes Prediction App (Logistic Regression)

**Short:** A Logistic Regression based diabetes risk predictor built on the Kaggle dataset.  
**Author:** Pasan Dharmadasa

## Live Demo
https://diabetes-prediction-app-app-crhyntb6cejmznadmpe27q.streamlit.app

## Features
- Trained using `StandardScaler` + `LogisticRegression` pipeline
- Evaluated using accuracy and F1 score
- Interactive Streamlit UI with 21 health & lifestyle inputs
- Predict probability and class label

## Performance
- Accuracy: **0.7458**
- F1 Score: **0.7503**

## Files
- `app.py` — Streamlit application
- `diabetes_model.pkl` — saved sklearn pipeline (scaler + model)
- `notebook.ipynb` — training notebook with experiments
- `requirements.txt` — dependencies for deployment

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py


<img width="1020" height="765" alt="Screenshot 2025-11-17 153215" src="https://github.com/user-attachments/assets/9fb3e991-29a0-482b-95c3-3baaaff8bb86" />
<img width="1023" height="798" alt="Screenshot 2025-11-17 153154" src="https://github.com/user-attachments/assets/a9a66d6e-c058-40b4-9227-90992b08571b" />
<img width="1014" height="645" alt="Screenshot 2025-11-17 153116" src="https://github.com/user-attachments/assets/8558d8ef-21c6-4971-b1c5-55f935fefae9" />
