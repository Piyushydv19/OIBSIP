#  Data Cleaning & Quality Analysis Dashboard

### By Piyush Yadav

## Overview

The Data Cleaning & Quality Analysis Dashboard is an interactive web application built using Python and Streamlit. It helps users analyze dataset quality, identify common data issues, clean datasets efficiently, and download the cleaned data for further analysis.

This project demonstrates essential data preprocessing and data quality management techniques commonly used in Data Analytics, Data Science, and Machine Learning workflows.

---

## Features

### 📊 Dataset Overview

* Upload CSV datasets
* View dataset shape and structure
* Display data types
* Preview dataset records

### 🔍 Data Quality Analysis

* Detect missing values
* Identify duplicate records
* Calculate dataset quality score
* Compare dataset quality before and after cleaning

### 🧹 Data Cleaning

* Missing value handling using Mean and Mode imputation
* Duplicate row removal
* Text standardization
* Data consistency improvements

### 📈 Outlier Detection

* IQR-based outlier detection
* Interactive boxplot visualization
* Outlier count by numeric column

### 📋 Cleaning Report

* Automatically generated cleaning summary
* Dataset statistics before and after cleaning
* Downloadable cleaning report

### 📥 Export Options

* Download cleaned dataset as CSV
* Download cleaned dataset as Excel
* Download cleaning report

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* OpenPyXL

---

## Project Structure

```text
data-cleaning-project/
│
├── app.py
├── utils.py
├── requirements.txt
├── README.md
│
└── datasets/
```

---

## Installation

### Clone Repository

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

The application will start locally and open in your browser.

---

## Workflow

```text
Upload Dataset
       ↓
Analyze Dataset
       ↓
Handle Missing Values
       ↓
Remove Duplicates
       ↓
Standardize Data
       ↓
Detect Outliers
       ↓
Generate Report
       ↓
Download Cleaned Dataset
```

---

## Key Concepts Implemented

* Data Integrity
* Missing Data Handling
* Duplicate Removal
* Data Standardization
* Outlier Detection
* Data Quality Assessment
* Data Visualization

---

## Learning Outcomes

This project demonstrates practical skills in:

* Data Cleaning
* Data Preprocessing
* Exploratory Data Analysis (EDA)
* Dashboard Development
* Python Programming
* Data Visualization
* Streamlit Application Development

---

## Future Improvements

* Automatic date format standardization
* Advanced outlier treatment methods
* PDF report generation
* Machine Learning based anomaly detection
* Database integration
* Multi-file support

---

## Author

**Piyush Yadav**

B.Tech (IoT) Student
Madhav Institute of Technology & Science (MITS), Gwalior

---

⭐ If you found this project useful, consider giving the repository a star.
