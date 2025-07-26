#  Rainfall Prediction Using Logistic Regression, Adaboost And Random Forest Algorithms

This project aims to predict rainfall using machine learning techniques such as Logistic Regression, AdaBoost, and Random Forest. The dataset contains various meteorological features, and the goal is to classify whether rainfall occurs based on those features.

---

##  Dataset

- **File:** `Rainfall.csv`
- The dataset contains various meteorological readings and a target column `rainfall` indicating rainfall occurrence.
- Missing values were handled using mean imputation.
- Categorical labels were encoded for model compatibility.

---

##  Models Used

1. **Logistic Regression**
   - Used for baseline binary classification.
   - Evaluated using accuracy, confusion matrix, and classification report.

2. **AdaBoost Classifier**
   - Used ensemble learning with 50 estimators.
   - Demonstrated improved generalization.

3. **Random Forest Classifier**
   - Used 100 trees with a max depth of 5.
   - Handled overfitting with `oob_score=True` and parallel training (`n_jobs=-1`).

---

##  Evaluation Metrics

- Accuracy Score.
- Classification Report (Precision, Recall, F1-Score).
- Confusion Matrix Visualization.

---

##  Project Workflow

1. **Data Loading & Cleaning**
   - Removed extra whitespace from column names.
   - Filled missing values using mean.
   - Encoded categorical variables.

2. **Exploratory Data Analysis (EDA)**
   - Used distribution plots and boxplots for numerical features.
   - Heatmap for correlation analysis.

3. **Train-Test Split**
   - Used 70-15-15 split (Train, Validation, Test).

4. **Model Training**
   - Each model was trained on the processed dataset.

5. **Visualization**
   - Confusion matrix and bar plots for performance analysis.

---

##  Sample Outputs

You can add screenshots of:
- Model Accuracy & Confusion Matrices
- Rainfall distribution bar chart
- EDA visualizations

---

##  Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

##  File Structure

- `rainfall_prediction_.py` - Main notebook/script with the entire code.
- `Rainfall.csv` - Input dataset file.

---

##  How to Run

1. Clone the repository or upload files to your Colab or local Python environment.
2. Ensure required libraries are installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
