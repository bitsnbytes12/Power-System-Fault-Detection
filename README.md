#  Power System Fault Detection and Classification

A Machine Learning-based web application that detects and classifies power system faults using operational and environmental parameters. The project was developed as part of the **Edunet Foundation – IBM SkillsBuild Internship Program** using **IBM Cloud Lite** and **IBM Cloud Object Storage**.

---

##  Problem Statement

**Problem Statement 41 – Power System Fault Detection and Classification**

Develop a machine learning model capable of detecting and classifying faults in a power distribution system to support faster fault diagnosis and improve grid reliability.

---

##  Features

* Fault type prediction using Machine Learning
* Data preprocessing and feature engineering
* Exploratory Data Analysis (EDA)
* Random Forest-based fault classification
* Flask web application for real-time predictions
* IBM Cloud Lite integration
* IBM Cloud Object Storage for dataset management

---

##  Technologies Used

* **Python**
* **Flask**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **IBM Cloud Lite**
* **IBM Cloud Object Storage**
* **Jupyter Notebook**

---

##  IBM Cloud Services Used

* **IBM Cloud Lite** – Cloud environment for project development.
* **IBM Cloud Object Storage** – Storage and management of the project dataset.

---


## Dataset

**Source:** Kaggle – Power System Faults Dataset

The dataset includes electrical and operational parameters such as:

* Voltage
* Current
* Power Load
* Temperature
* Wind Speed
* Weather Condition
* Maintenance Status
* Component Health
* Fault Duration
* Downtime

**Target Variable:** Fault Type

---

##  Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Hyperparameter Tuning
7. Model Evaluation
8. Flask Application Deployment

---

##  Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

The **Random Forest Classifier** achieved the best overall performance and was selected as the final model.
