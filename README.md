# hate-speech-detection-logistic-regression
Logistic Regression model for detecting hate and offensive speech using NLP.
# 📌 Hate Speech Detection using Logistic Regression  

A machine learning project that detects whether a text/tweet contains **hate speech**, **offensive language**, or is **neutral**.  
This model uses **TF-IDF Vectorization** and **Logistic Regression** to classify text effectively.

---

## 🔧 Tech Stack  
- Python  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib, Seaborn  
- Natural Language Processing (NLP)

---

## 📂 Project Structure

---

## 📊 Dataset  
Dataset Used: **Hate Speech and Offensive Language Dataset**  

Labels:  
- **0 — Hate Speech**  
- **1 — Offensive Language**  
- **2 — Neutral / Normal**

---

## 🚀 How the Model Works  
### **1️⃣ Data Loading**  
Load CSV file with tweets and labels.

### **2️⃣ Data Visualization**  
Plot class distribution to understand dataset balance.

### **3️⃣ Preprocessing**  
- Encode labels  
- Train-test split  
- TF-IDF vectorization (5000 features)

### **4️⃣ Model Training**  
- Logistic Regression  
- Max iterations: **200**

### **5️⃣ Evaluation**  
- Accuracy  
- Classification Report  
- Confusion Matrix (Heatmap)

---

## 🧠 Full Project Code  
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Load Dataset
df = pd.read_csv("hate-speech-and-offensive-language.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ==============================
# Step 2: Data Visualization
# ==============================
plt.figure(figsize=(7, 4))
sns.countplot(x=df["class"])
plt.title("Class Distribution")
plt.xlabel("Class (0 = Hate, 1 = Offensive, 2 = Normal)")
plt.ylabel("Count")
plt.show()

# ==============================
# Step 3: Feature Selection
# ==============================
X = df['tweet']       # Input feature (tweets)
y = df['class']       # Target labels

# Step 4: Label Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 7: Train Model (Logistic Regression)
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# Step 8: Make Predictions
y_pred = model.predict(X_test_tfidf)

# ==============================
# Step 9: Evaluation
# ==============================
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
target_names = [str(c) for c in label_encoder.classes_]
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
