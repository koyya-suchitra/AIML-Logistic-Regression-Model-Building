# AIML-Logistic-Regression-Model-Building
Smart Screening for Smarter Healthcare
# 🩺 Breast Cancer Analysis using Logistic Regression

This project demonstrates a complete pipeline for **binary classification** using **Logistic Regression**, specifically applied to **breast cancer diagnosis**. It includes data preprocessing, model training, evaluation, and inference on new unseen data using a clean and interpretable approach.

The model helps classify tumors as **Malignant** or **Benign** based on input features extracted from medical imaging.

---

## 📁 Project Structure

```
├── logreg.ipynb                # Training and evaluation of Logistic Regression model
├── test_on_new_data.ipynb     # Inference on new test dataset using the trained model
├── oriiginaldata.csv        # (Optional) Breast cancer dataset used in training
├── README.md                  # Project documentation
├── newdata.csv                #5 new samples used
```

---

## ✅ Tasks Completed

### 1. **Dataset Selection: Breast Cancer Diagnosis**
* Used a publicly available breast cancer dataset (e.g., **Wisconsin Diagnostic Breast Cancer**).
* Binary target: `Malignant (1)` vs `Benign (0)`.

### 2. **Data Preprocessing**
* **Train/Test Split**: 80/20 ratio for training and evaluation.
* **Feature Scaling**: Applied `StandardScaler` to normalize the data.

### 3. **Model Training**
* Used `LogisticRegression` from `sklearn.linear_model`.
* Trained on standardized features to maximize convergence and interpretability.

### 4. **Model Evaluation**
* Computed key evaluation metrics:
  - Confusion Matrix
  - Accuracy
  - Precision, Recall, F1-score
  - ROC-AUC Score
* Plotted:
  - ROC Curve
  - Confusion Matrix Heatmap

### 5. **Threshold Tuning and Sigmoid Interpretation**
* Explained the **Sigmoid Function** and how it maps input to probabilities.
* Showed the impact of adjusting the **classification threshold** to optimize recall or precision based on medical priority.

---

## 🔍 Model Inference on New Data

* The trained **Logistic Regression** model and the **StandardScaler** were saved using the `pickle` module for future use.
* During inference, both the model and scaler were loaded using `pickle.load()`, ensuring consistency in preprocessing.
* The new input data was scaled using the same `StandardScaler` from training.
* Predictions were then generated using the reloaded model.
* Final outputs, including predicted labels were visualized and saved for analysis.
* 
---

## 🧠 Technologies Used

* Python 3.x
* scikit-learn
* pandas
* numpy
* matplotlib / seaborn
* Jupyter Notebook
* pickle

---

## 🚀 How to Run This Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-logreg.git
   cd breast-cancer-logreg
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```

3. **Open and run the notebooks**:
   - `logreg.ipynb` to train and evaluate the model
   - `test_on_new_data.ipynb` to test the model on unseen data

---

## 📌 Notes

* Ensure that new input data has the **same features** and is **standardized** using the scaler used during training.
* You can modify the decision threshold to optimize for **recall (reducing false negatives)** in clinical scenarios.

---

## 📜 License

This project is developed for **educational and research purposes** only. Always consult medical professionals before making any healthcare-related decisions.

---

## ❤️ Acknowledgements

* Dataset inspired by the [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

```


