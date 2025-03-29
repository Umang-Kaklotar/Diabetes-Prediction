# 🏥 Diabetes Prediction using Logistic Regression

## 📌 Overview

<p>This project implements <strong>Logistic Regression</strong> from scratch to predict diabetes based on medical data. The approach follows a structured <strong>machine learning pipeline</strong>, including <strong>data preprocessing, feature scaling, model training, evaluation, and prediction</strong>. The model is optimized using <strong>gradient descent</strong> to minimize the cost function.</p>

## 📂 Project Structure

<ul>
  <li><strong>DiabetesPrediction.ipynb</strong> - Jupyter Notebook with the complete model implementation.</li>
  <li><strong>diabetes.csv</strong> - Dataset used for training and testing.</li>
  <li><strong>README.md</strong> - Documentation for the project.</li>
</ul>

## 🛠 Approach

<h3>1️⃣ Data Preprocessing</h3>
<ul>
  <li>Load the <strong>Pima Indians Diabetes Dataset</strong>.</li>
  <li>Check for missing values and outliers.</li>
  <li>Perform <strong>feature scaling</strong> using <code>StandardScaler</code> to normalize data.</li>
</ul>

<h3>2️⃣ Model Implementation</h3>
<ul>
  <li>Implement <strong>Logistic Regression</strong> using NumPy without external ML libraries.</li>
  <li>Define key functions:
    <ul>
      <li><strong>Sigmoid Function</strong> for probability computation.</li>
      <li><strong>Cost Function</strong> to measure prediction error.</li>
      <li><strong>Gradient Descent Optimization</strong> for parameter updates.</li>
    </ul>
  </li>
</ul>

<h3>3️⃣ Model Training</h3>
<ul>
  <li>Split data into <strong>80% training</strong> and <strong>20% testing</strong> using <code>train_test_split()</code>.</li>
  <li>Train the model using <strong>gradient descent</strong> with a defined learning rate and iterations.</li>
  <li>Monitor <strong>cost reduction</strong> over iterations to ensure convergence.</li>
</ul>

<h3>4️⃣ Model Evaluation</h3>
<ul>
  <li>Evaluate using:
    <ul>
      <li><strong>Accuracy</strong></li>
      <li><strong>Precision, Recall, and F1-score</strong></li>
      <li><strong>Confusion Matrix</strong></li>
      <li><strong>Loss Curve Visualization</strong></li>
    </ul>
  </li>
  <li>Compare results with <strong>standard logistic regression from Scikit-Learn</strong>.</li>
</ul>

<h3>5️⃣ Making Predictions</h3>
<ul>
  <li>Test the model on unseen patient data.</li>
  <li>Standardize new input features before prediction.</li>
  <li>Classify as <strong>Diabetic (1) / Non-Diabetic (0)</strong>.</li>
</ul>

## 📊 Dataset

<p>The dataset used in this project is the <strong>PIMA Indians Diabetes Dataset</strong>. It contains medical data for female patients and is used to predict the likelihood of diabetes based on specific features.</p>

<h3>Dataset Features:</h3>
<ul>
  <li><strong>Pregnancies</strong>: Number of pregnancies</li>
  <li><strong>Glucose</strong>: Plasma glucose concentration</li>
  <li><strong>BloodPressure</strong>: Diastolic blood pressure (mm Hg)</li>
  <li><strong>SkinThickness</strong>: Triceps skinfold thickness (mm)</li>
  <li><strong>Insulin</strong>: 2-Hour serum insulin (mu U/ml)</li>
  <li><strong>BMI</strong>: Body mass index (weight in kg / height in m²)</li>
  <li><strong>DiabetesPedigreeFunction</strong>: Diabetes pedigree function (a measure of genetic risk)</li>
  <li><strong>Age</strong>: Age of the patient</li>
  <li><strong>Outcome</strong>: 1 if the patient has diabetes, 0 otherwise</li>
</ul>

## 🔧 Technologies Used

<ul>
  <li><strong>Python</strong></li>
  <li><strong>NumPy</strong></li>
  <li><strong>Pandas</strong></li>
  <li><strong>Matplotlib & Seaborn</strong></li>
  <li><strong>Scikit-Learn</strong> (for dataset handling & evaluation)</li>
</ul>

## 📊 Performance Metrics

<ul>
  <li><strong>Train Accuracy:</strong> ~77.04%</li>
  <li><strong>Test Accuracy:</strong> ~73.38%</li>
  <li><strong>Precision, Recall, and F1-score</strong> for deeper analysis.</li>
</ul>

## 📥 Installation

<p>Install dependencies:</p>

<pre><code>pip install numpy pandas matplotlib seaborn scikit-learn</code></pre>

## 🎯 Usage

<ul>
  <li>Open <code>DiabetesPrediction.pynb</code> in Jupyter Notebook.</li>
  <li>Execute cells sequentially to preprocess data, train the model, and evaluate performance.</li>
  <li>Modify <code>new_patient</code> array to test with new data.</li>
</ul>

## 🚀 Future Enhancements

<ul>
  <li>Implement <strong>Regularization (L1/L2)</strong> to reduce overfitting.</li>
  <li>Use <strong>Feature Engineering</strong> for better insights.</li>
  <li>Compare performance with <strong>Deep Learning models (e.g., Neural Networks)</strong>.</li>
</ul>
