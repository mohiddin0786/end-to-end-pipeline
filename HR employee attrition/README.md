# 🧑‍💼 Employee Attrition Prediction – End-to-End Machine Learning Pipeline

## 📌 Project Overview
Employee attrition is a critical challenge for organizations, impacting productivity, hiring costs, and workforce stability.  
This project builds an **end-to-end machine learning pipeline** to analyze and predict employee attrition using structured HR data.

The focus of this project is not only on model performance, but also on:
- clean exploratory data analysis
- meaningful business insights
- modular and reusable pipeline design
- reproducible model evaluation

---

## 📂 Project Structure

hr-employee-attrition/
├── data/
│ └── HR_attrition_data.csv
├── notebooks/
│ └── exploration.ipynb
├── src/
│ ├── Data_Loader.py
│ ├── preprocesser.py
│ ├── Train.py
│ ├── Evaluation.py
│ └── pipeline.py
├── .gitignore
└── README.md


---

## 📊 Exploratory Data Analysis (EDA)
Exploratory Data Analysis was conducted to understand the underlying factors contributing to employee attrition.

### Key insights from EDA:
- Employees who left the company generally exhibit **lower satisfaction levels**
- **Higher workload** (average monthly working hours) is associated with increased attrition
- Attrition rates vary across departments, indicating department-specific influences
- Employees with **low or medium salaries and no recent promotions** are more likely to leave

These insights guided feature selection and model choice.

---

## 🛠️ Data Preprocessing
- Categorical variables were encoded using **one-hot encoding**
- Target variable: `left`
  - `0` → Employee stayed
  - `1` → Employee left
- Dataset was split into **training and test sets**
- Class imbalance was handled during model training

---

## 🤖 Models Trained
Two baseline classification models were trained and evaluated:

### 1️⃣ Logistic Regression
- Used as an **interpretable baseline model**
- Helps understand linear relationships between features and attrition
- Provides transparency but limited performance on minority class

### 2️⃣ Random Forest Classifier
- Captures **non-linear relationships** between features
- Handles complex feature interactions effectively
- Uses class balancing to improve attrition detection

---

## 📈 Model Evaluation & Results

### 🔹 Logistic Regression Performance

**Training Data**
- Accuracy: **79%**
- Attrition class (1) recall: **32%**
- Performs reasonably as a baseline but struggles with minority class detection

**Test Data**
- Accuracy: **79%**
- Attrition class (1) recall: **34%**
- Consistent generalization with limited sensitivity to attrition cases

➡️ *Logistic Regression offers interpretability but underperforms in identifying employees likely to leave.*

---

### 🔹 Random Forest Performance

**Training Data**
- Accuracy: **99%**
- Strong precision and recall for both classes

**Test Data**
- Accuracy: **98%**
- Attrition class (1) recall: **94%**
- F1-score (attrition class): **0.96**

➡️ *Random Forest significantly outperforms Logistic Regression, particularly in detecting employees at risk of attrition.*

---

## 🧠 Key Observations
- Employee satisfaction and workload are strong predictors of attrition
- Tree-based models outperform linear models for this problem
- Addressing class imbalance is crucial for attrition prediction
- Accuracy alone is insufficient; **recall for the attrition class** is a critical metric

---

## 🏁 Conclusion
This project demonstrates how a structured end-to-end machine learning pipeline can effectively analyze and predict employee attrition.

While Logistic Regression provides a transparent baseline, Random Forest delivers substantially better performance in identifying at-risk employees. The modular pipeline design ensures the solution is extensible, reproducible, and aligned with real-world analytics workflows.

---

## ▶️ How to Run
1. Clone the repository
2. Place the dataset inside the `data/` directory
3. Run the pipeline using:
```bash
python src/pipeline.py
