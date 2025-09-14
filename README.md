# 🔬 Thyroid Cancer Recurrence Prediction Project

## 📌 Objective  
Can we predict if a thyroid cancer survivor will experience a recurrence using clinical and pathological data?

This project is my **end-to-end journey** tackling a real-world healthcare challenge using machine learning.

The goal: build a **robust and interpretable model** that helps clinicians identify patients at higher risk of recurrence.

> ⚕️ Early detection of recurrence risk can lead to **better monitoring, personalized care, and improved patient outcomes**.

---

## 🚀 Live Demo  
Try out the app here:  
[**Live Thyroid Recurrence Predictor**](https://thyroid-cancer-recurrence-prediction-rsybno6wfty5wyb2pc2meg.streamlit.app/)  

---

## 🛠️ Tech Stack & Libraries  

| Technology        | Description                                        |
|--------------------|----------------------------------------------------|
| **Python**         | Core language for data analysis & modeling        |
| **Pandas & NumPy** | Efficient data manipulation & numerical operations |
| **Scikit-Learn**   | ML modeling (Logistic Regression, Random Forest)   |
| **Matplotlib & Seaborn** | Data visualization during EDA               |
| **SHAP**           | Model interpretability & feature importance        |
| **Streamlit**      | Interactive dashboard & deployment                 |
| **Joblib**         | Saving & loading trained models & scalers          |

---

```plaintext
Thyroid-Cancer-Recurrence-Prediction/
├── app.py                          # Streamlit web application
├── predictions.py                  # Core prediction logic
├── tuned_random_forest_model.pkl   # Final trained ML model
├── scaler.pkl                      # Scaler for feature normalization
├── requirements.txt                # Project dependencies
├── THYROID CANCER PREDICTION.ipynb # Notebook: EDA + model training
├── assets/                         # Images (background, plots, etc.)

---

## 📈 Model Performance  

After hyperparameter tuning (RandomizedSearchCV) with 5-fold cross-validation, the final **Random Forest** model achieved:

- ✅ **Accuracy:** 97% (on unseen test set)  
- ✅ **ROC-AUC:** 0.988 (excellent class separation)  

The metrics show good balance between precision and recall, making the model reliable in identifying recurrence risk.

---

## 🧠 Model Interpretability with SHAP  

To ensure the model isn’t a black box, I used **SHAP (SHapley Additive exPlanations)** for interpreting feature importance.

### 🔑 Key Insights:
1. **Response to initial treatment** — the most powerful predictor  
2. **Risk category & lymph node involvement** (N stage, Adenopathy) are also major contributors  
3. These insights align with clinical intuition, enhancing trust in the model’s predictions

---

## 🔧 How to Run Locally  

1️⃣ **Clone the repository**  
```bash
git clone https://github.com/Swaransh-Mishra/Thyroid-Cancer-Recurrence-Prediction.git
cd Thyroid-Cancer-Recurrence-Prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py

🙌 Closing Thoughts

This project was more than a coding exercise — it was an exploration into how machine learning can augment healthcare decisions. By predicting recurrence risk in thyroid cancer survivors, the model demonstrates the potential of AI in providing actionable, trustworthy insights for clinicians and patients alike.

⚕️ While this is not a replacement for medical judgment, it is a step toward data-driven healthcare that supports earlier interventions, personalized monitoring, and better outcomes.


## 👤 Author  
**Swaransh Mishra**  

- GitHub: [@Swaransh-Mishra](https://github.com/Swaransh-Mishra)  
- LinkedIn: [Swaransh Mishra](https://www.linkedin.com/in/swaransh-mishra-a85123258/)  
