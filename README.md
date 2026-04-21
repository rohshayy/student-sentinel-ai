# Student Sentinel AI
### **Dual-Model Engine for GPA Forecasting & Latent Dropout Risk Detection**

## **Executive Summary**
Student Sentinel AI is a proactive intervention tool designed for higher education institutions. By utilizing digital proxies—such as LMS engagement and Wi-Fi logs—alongside psychometric indicators, the system predicts academic decline before it manifests. 

Unlike reactive systems, this engine provides a dual-output: a **continuous performance forecast (GPA)** and a **probabilistic binary risk alert (Dropout Risk)**.

## **Technical Architecture**

### **1. Dual-Task Modeling**
The system processes multi-dimensional data through an optimized Scikit-Learn Pipeline:
* **Regression Path:** Utilizes Linear Regression to estimate specific GPA outcomes for the upcoming semester.
* **Classification Path:** Employs Logistic Regression with a **Sigmoid Activation Function** to calculate the probability of a student crossing the "At-Risk" threshold.

### **2. Feature Engineering & Distribution Modeling**
To reflect real-world academic environments, features are modeled via specific statistical distributions:
* **Engagement:** Modeled via **Poisson Distributions** (LMS clicks) and **Gaussian Distributions** (Library hours).
* **Stressors:** Modeled via **Uniform Distributions** (Commute distance).
* **The Interest Gap:** A custom latent variable calculating the delta between elective performance and core-major grades to identify misalignment between student interests and their chosen field.

### **3. Mathematical Engineering**
* **Z-Score Standardization:** Implemented `StandardScaler` to ensure features on disparate scales contribute proportionally to Gradient Descent optimization.
* **Stochastic Noise Modeling:** Incorporates **Gaussian noise ($\epsilon$)** into training simulations to account for human behavioral unpredictability, improving model generalization.
* **Risk Stratification:** Leverages `predict_proba` to rank interventions by mathematical certainty, moving beyond binary labels to nuanced risk scores.

## **Key Performance Metrics**
* **Predictive Accuracy:** Captured ~67% of the variance ($R^2$) in performance, demonstrating strong correlation despite simulated environmental noise.
* **Actionable Intelligence:** The system generates an automated `Priority_Counseling_List.csv`, sorting at-risk students by failure probability.

## **Technical Stack**
* **Language:** Python
* **Libraries:** Scikit-Learn, NumPy, Pandas
* **Environment:** PyCharm
