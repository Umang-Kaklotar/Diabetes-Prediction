🏥 Diabetes Prediction using Logistic Regression

Overview

This project implements Logistic Regression from scratch to predict diabetes based on medical data. The approach follows a structured machine learning pipeline, including data preprocessing, feature scaling, model training, evaluation, and prediction. The model is optimized using gradient descent to minimize the cost function.

📂 Project Structure

DiabetesPrediction.ipynb - Jupyter Notebook with the complete model implementation.

diabetes.csv - Dataset used for training and testing.

README.md - Documentation for the project.

🛠 Approach

1️⃣ Data Preprocessing

Load the Pima Indians Diabetes Dataset.

Check for missing values and outliers.

Perform feature scaling using StandardScaler to normalize data.

2️⃣ Model Implementation

Implement Logistic Regression using NumPy without external ML libraries.

Define key functions:

Sigmoid Function for probability computation.

Cost Function to measure prediction error.

Gradient Descent Optimization for parameter updates.

3️⃣ Model Training

Split data into 80% training and 20% testing using train_test_split().

Train the model using gradient descent with a defined learning rate and iterations.

Monitor cost reduction over iterations to ensure convergence.

4️⃣ Model Evaluation

Evaluate using:

Accuracy

Precision, Recall, and F1-score

Confusion Matrix

Loss Curve Visualization

Compare results with standard logistic regression from Scikit-Learn.

5️⃣ Making Predictions

Test the model on unseen patient data.

Standardize new input features before prediction.

Classify as Diabetic (1) / Non-Diabetic (0).

📊 Dataset

The dataset used in this project is the PIMA Indians Diabetes Dataset. It contains medical data for female patients and is used to predict the likelihood of diabetes based on specific features.

Dataset Features:

Pregnancies: Number of pregnancies

Glucose: Plasma glucose concentration

BloodPressure: Diastolic blood pressure (mm Hg)

SkinThickness: Triceps skinfold thickness (mm)

Insulin: 2-Hour serum insulin (mu U/ml)

BMI: Body mass index (weight in kg / height in m²)

DiabetesPedigreeFunction: Diabetes pedigree function (a measure of genetic risk)

Age: Age of the patient

Outcome: 1 if the patient has diabetes, 0 otherwise

🔧 Technologies Used

Python

NumPy

Pandas

Matplotlib & Seaborn

Scikit-Learn (for dataset handling & evaluation)

📊 Performance Metrics

Train Accuracy: ~XX%

Test Accuracy: ~XX%

Precision, Recall, and F1-score for deeper analysis.

📥 Installation 

Install dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn

🎯 Usage

Open DiabetesPrediction.ipynb in Jupyter Notebook.

Execute cells sequentially to preprocess data, train the model, and evaluate performance.

Modify new_patient array to test with new data.

🚀 Future Enhancements

Implement Regularization (L1/L2) to reduce overfitting.

Use Feature Engineering for better insights.

Compare performance with Deep Learning models (e.g., Neural Networks).
