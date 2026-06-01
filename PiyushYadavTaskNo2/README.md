# 👥 Customer Segmentation Analysis Dashboard

An end-to-end Data Analytics and Machine Learning project that segments customers into meaningful groups using **K-Means Clustering**. The dashboard enables businesses to understand customer behavior, identify high-value customers, and create targeted marketing strategies.

---

## 📌 Project Overview

Customer segmentation is a crucial business technique used to divide customers into distinct groups based on purchasing behavior, demographics, and engagement patterns.

This project applies:

* Data Cleaning & Preprocessing
* Feature Engineering
* RFM Analysis (Recency, Frequency, Monetary)
* K-Means Clustering
* PCA Visualization
* Interactive Business Dashboard

The entire workflow is deployed as a **Streamlit application** with rich visualizations and downloadable insights.

---

## 🚀 Key Features

### 📊 Exploratory Data Analysis

* Dataset overview and quality checks
* Missing value analysis
* Income, age, and spending distributions
* Correlation heatmap
* Customer behavior exploration

### 🎯 Customer Segmentation

* Automatic customer grouping using K-Means
* Cluster comparison and evaluation
* Segment size analysis
* PCA-based cluster visualization

### 📈 Cluster Optimization

* Elbow Method
* Silhouette Score Analysis
* Davies-Bouldin Index
* Optimal cluster selection

### 📋 Segment Profiling

* Average income by segment
* Spending behavior comparison
* Purchase channel preferences
* Family and demographic insights
* Radar chart visualization

### 💡 Business Insights

* High-value customer identification
* Churn-risk customer detection
* Marketing recommendations
* Customer retention strategies
* Cross-selling opportunities

---

## 🛠 Technology Stack

| Category                 | Tools         |
| ------------------------ | ------------- |
| Programming              | Python        |
| Dashboard                | Streamlit     |
| Data Processing          | Pandas, NumPy |
| Machine Learning         | Scikit-Learn  |
| Visualization            | Plotly        |
| Dimensionality Reduction | PCA           |
| Clustering               | K-Means       |

---

## 📂 Project Structure

```text
Customer-Segmentation-Analysis/
│
├── app.py
├── requirements.txt
├── README.md
│
└── utils/
    ├── data_prep.py
    ├── clustering.py
    └── charts.py
```

---

## 🔄 Project Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Customer Behavior Analysis
5. Feature Scaling
6. K-Means Clustering
7. Cluster Evaluation
8. PCA Visualization
9. Segment Profiling
10. Business Recommendations

---

## 📊 Features Used for Clustering

The clustering model uses the following customer attributes:

* Income
* Age
* Recency
* Total Spend
* Total Purchases
* Average Order Value
* Number of Children
* Campaign Acceptance Rate
* Online Purchase Ratio
* Catalog Purchase Ratio
* Customer Seniority

These features help identify customers with similar purchasing patterns and engagement levels.

---

## 📈 Dashboard Pages

| Dashboard Page             | Description                                 |
| -------------------------- | ------------------------------------------- |
| Overview & EDA             | Dataset exploration and customer statistics |
| Optimal Clusters           | Elbow curve and clustering metrics          |
| Segmentation Results       | Cluster assignments and PCA visualization   |
| Segment Profiles           | Detailed customer segment characteristics   |
| Insights & Recommendations | Actionable business strategies              |

---

## 🎯 Business Value

This project helps organizations:

* Improve customer retention
* Increase marketing effectiveness
* Personalize customer experiences
* Identify premium customers
* Optimize campaign targeting
* Enhance revenue generation

---

## ▶️ Installation

### Clone Repository

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Dashboard

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```


## 📚 Machine Learning Concepts Used

* Data Preprocessing
* Feature Engineering
* Standardization
* K-Means Clustering
* Elbow Method
* Silhouette Analysis
* PCA (Principal Component Analysis)
* Customer Segmentation

---

## 👨‍💻 Author

**Piyush Yadav**

Data Analytics & Machine Learning Enthusiast

Project developed as part of the **Oasis Infobyte Data Analytics Internship (OIBSIP)**.
