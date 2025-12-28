🏡 Bangalore House Price Predictor

A Machine Learning web application that estimates real estate prices in Bangalore based on location, square footage, bathrooms, and BHK.

🚀 How It Works
1. **Data Cleaning:** Processed raw real estate data (`cleaned_train.csv`).
2. **Model Training:** Built a **Random Forest Regressor** using `house_price_analysis.py`.
3. **Web App:** Created an interactive dashboard using **Streamlit** (`app.py`).

🛠️ Tech Stack
* **Python** (Pandas, NumPy)
* **Machine Learning** (Scikit-Learn, Random Forest)
* **Frontend** (Streamlit)

📦 How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   
