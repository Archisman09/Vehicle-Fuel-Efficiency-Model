### **Car MPG Predictor 🚗**  
A **Streamlit-based web app** that predicts **Miles Per Gallon (MPG)** for a car based on its specifications using a **Machine Learning model** trained on historical car data from 1970-1982.

---

## 📌 **Features**
✅ Predicts **MPG** based on user inputs.  
✅ Supports **batch predictions** from CSV files.  
✅ Uses **scikit-learn model** for regression.  
✅ Provides **visualizations** for insights.  
✅ Hosted via **Streamlit** for easy accessibility.  

---

## 🚀 **How to Run Locally**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Streamlit App**
```bash
streamlit run app.py
```

---

## 📂 **Project Structure**
```
📂 Car-MPG-Predictor
│-- app.py               # Main Streamlit app
│-- model.pkl            # Trained ML model (Pickle file)
│-- scaler.pkl           # StandardScaler for preprocessing
│-- feature_names.pkl    # Expected feature names for input
│-- requirements.txt     # Required Python packages
│-- MPG ML Model and EDA.ipynb  # Exploratory Data Analysis & Model Training Notebook
```

---

## 🖥️ **Web App Interface**
### **🔹 Input Section**
- Select **cylinders, displacement, horsepower, weight, acceleration, model year, and origin**.
- Click **Predict MPG** to see results.

### **📁 Batch Prediction**
- Upload a **CSV file** with multiple car specifications.
- The app will predict **MPG** for all entries.

---

## 📊 **Machine Learning Model**
- Trained on **historical car data (1970-1982)**.
- Uses **scikit-learn's regression model**.
- Feature scaling applied via **StandardScaler**.

---

## 🌐 **Deploying on Streamlit Cloud**
1. Upload your project to **GitHub**.
2. Visit **[Streamlit Cloud](https://share.streamlit.io/)**.
3. Select your **repository** and **branch**.
4. Set `app.py` as the main file.
5. Click **"Deploy"**.

---

## 🛠 **Tech Stack**
- **Frontend:** Streamlit  
- **Backend:** Scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  

---

## ✨ **Contributions**
Contributions are welcome! Feel free to submit **issues** or **pull requests**.

---

## 📜 **License**
This project is licensed under the **MIT License**.

---
