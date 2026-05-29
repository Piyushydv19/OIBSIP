# 📱 AppVision: Android Market Analytics Platform

## 🚀 Project Overview

**AppVision** is a comprehensive Data Analytics Dashboard built using **Python, Streamlit, Pandas, and Plotly** to analyze Google Play Store applications and user reviews.

The platform transforms raw Play Store data into actionable insights through interactive visualizations, sentiment analysis, and market intelligence dashboards.

This project helps identify:

* Most popular app categories
* User rating trends
* Installation patterns
* Pricing strategies
* User sentiment from reviews
* Market opportunities for new app launches

---

## 🎯 Objectives

* Clean and preprocess Google Play Store datasets
* Perform Exploratory Data Analysis (EDA)
* Analyze app ratings, installs, pricing, and categories
* Extract user sentiment from app reviews
* Build an interactive dashboard for business insights
* Generate recommendations based on market trends

---

## 📊 Dashboard Modules

### 🏠 Overview Dashboard

* Total Apps
* Total Categories
* Average Rating
* Total Reviews
* Total Installs
* Free vs Paid Apps Distribution
* Data Quality Report

### 📂 Category Analysis

* Top Categories by App Count
* Average Rating by Category
* Total Installs by Category
* Category Performance Comparison

### ⭐ Ratings Analysis

* Rating Distribution
* Top Rated Categories
* Lowest Rated Categories
* Rating vs Reviews Analysis
* Rating vs Installs Analysis

### 📥 Installs & Popularity Analysis

* Most Installed Applications
* Most Reviewed Applications
* Popular Categories
* Installs vs Reviews Relationship

### 💰 Pricing Analysis

* Free vs Paid Apps
* Price Distribution
* Average Price by Category
* Revenue Opportunity Analysis

### 📦 App Size Analysis

* App Size Distribution
* Size vs Rating
* Size vs Installs
* Category-wise Size Comparison

### 💬 Sentiment Analysis

* Positive Reviews Analysis
* Negative Reviews Analysis
* Neutral Reviews Analysis
* Word Cloud Visualization
* Sentiment Distribution Dashboard

### 💡 Insights & Recommendations

* Best Category to Launch Apps
* Highest Rated Categories
* Highest Install Volume Categories
* Least Competitive Categories
* Free vs Paid Performance Comparison
* Top 10 Overall Apps Ranking

---

## 📈 Key Features

✅ Interactive Streamlit Dashboard

✅ Dynamic Filtering System

✅ CSV Dataset Upload Support

✅ Automated Data Cleaning

✅ Sentiment Analysis using TextBlob

✅ Download Cleaned Dataset

✅ Download Insights Report

✅ Responsive Visualizations

✅ Dark Theme Compatible UI

---

## 🛠️ Technology Stack

| Category             | Tools              |
| -------------------- | ------------------ |
| Programming Language | Python             |
| Dashboard Framework  | Streamlit          |
| Data Processing      | Pandas, NumPy      |
| Visualization        | Plotly, Matplotlib |
| Sentiment Analysis   | TextBlob           |
| Word Cloud           | WordCloud          |
| Version Control      | Git & GitHub       |

---

## 📂 Project Structure

```text
AppVision/
│
├── app.py
├── requirements.txt
├── README.md
│
├── datasets/
│   ├── googleplaystore.csv
│   └── googleplaystore_user_reviews.csv
│
└── utils/
    ├── __init__.py
    ├── data_cleaning.py
    ├── sentiment_analysis.py
    └── visualizations.py
```

---

## 📥 Dataset

### Google Play Store Dataset

Contains:

* App Name
* Category
* Rating
* Reviews
* Installs
* Price
* Size
* Content Rating

### User Reviews Dataset

Contains:

* User Reviews
* Sentiment
* Sentiment Polarity
* Sentiment Subjectivity

Source:
https://www.kaggle.com/datasets/lava18/google-play-store-apps

---

## ⚙️ Installation & Setup

### Clone Repository

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download TextBlob Corpora

```bash
python -m textblob.download_corpora
```

### Run Application

```bash
streamlit run app.py
```


## 📊 Business Insights Generated

The dashboard automatically identifies:

* High-growth app categories
* User preference patterns
* Revenue opportunities
* Competitive market segments
* Customer sentiment trends
* Potential categories for new app development

---

## 🎓 Learning Outcomes

Through this project, the following skills were developed:

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Data Visualization
* Sentiment Analysis
* Dashboard Development
* Business Intelligence
* Streamlit Deployment
* Git & GitHub Project Management

---

## 🔮 Future Enhancements

* Machine Learning based Rating Prediction
* App Success Prediction Model
* Recommendation Engine
* Real-Time Play Store Data Integration
* AI-Powered Market Insights
* User Authentication System

---

## 👨‍💻 Author

**Piyush Yadav**

B.Tech (IoT) | Data Analytics & Python Enthusiast

---

⭐ If you found this project useful, consider giving it a star on GitHub.

