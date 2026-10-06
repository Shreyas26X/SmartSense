# SmartSense

## AI-Powered Customer Segmentation and Personalized Campaign Recommendation

SmartSense is a machine-learning-based customer segmentation system that analyzes customer purchasing behaviour and groups customers into meaningful segments. The system uses transaction data to identify behavioural patterns and provides personalized marketing campaign recommendations for each segment.

The project combines **Python, Flask, Pandas, NumPy and Scikit-learn** to provide an end-to-end data science and web application workflow.

---

## Project Overview

Businesses often have large amounts of customer transaction data but may find it difficult to identify different customer behaviours and target them effectively.

SmartSense addresses this problem by:

- Processing customer transaction data
- Performing feature engineering
- Calculating customer **Recency** and **Average Order Value (AOV)**
- Scaling numerical features
- Applying **K-Means clustering**
- Identifying customer segments
- Providing personalized campaign recommendations
- Visualizing segmentation results through a web dashboard
- Allowing processed results to be downloaded

---

## Objectives

The main objectives of SmartSense are:

1. To analyze customer purchasing behaviour using transaction data.
2. To identify meaningful customer groups using unsupervised machine learning.
3. To apply K-Means clustering for customer segmentation.
4. To transform raw transaction data into useful behavioural features.
5. To provide business-oriented names and recommendations for each segment.
6. To provide an interactive web interface for analysing customer data.

---

## Methodology

```text
Customer Transaction Data
          |
          v
     Data Processing
          |
          v
    Feature Engineering
          |
          v
 Recency & Average Order Value
          |
          v
     Feature Scaling
          |
          v
     K-Means Clustering
          |
          v
    Customer Segmentation
          |
          v
 Campaign Recommendations
          |
          v
     Web Visualization
```

---

## Features

### 1. Data Processing

The system accepts customer transaction data in CSV format and processes the required fields.

### 2. Feature Engineering

The project derives behavioural features such as:

- Recency
- Total Price
- Average Order Value

### 3. Feature Scaling

`MinMaxScaler` is used to normalize the numerical features before applying clustering.

### 4. K-Means Clustering

K-Means clustering is used to divide customers into **four behavioural clusters**.

### 5. Customer Segmentation

The generated clusters are mapped to meaningful customer categories:

| Segment | Recommended Campaign |
|---|---|
| High-Value Loyal Customers | Exclusive Membership & VIP Rewards |
| Seasonal Buyers | Seasonal Promotions & Personalized Offers |
| Low-Value Frequent Buyers | Discount Coupons & Loyalty Programs |
| Lost Customers | Standard Re-engagement Emails |

### 6. Visualization

The application provides visual representations of:

- Customer segment distribution
- Customer spending behaviour
- Cluster relationships
- Seasonal distribution

### 7. Downloadable Results

Processed customer segmentation results can be downloaded for further analysis.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Flask | Web application framework |
| Pandas | Data processing and analysis |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning and preprocessing |
| K-Means | Customer clustering algorithm |
| Joblib | Model and scaler serialization |
| HTML/CSS | Web interface |

---

## Project Structure

```text
SmartSense/
|
├── app.py
├── train_model.ipynb
├── requirements.txt
|
├── templates/
│   └── ...
|
├── static/
│   └── ...
|
├── dataset/
│   └── ...
|
├── kmeans_model.joblib
├── scaler.joblib
|
└── README.md
```

> The exact file structure may vary depending on the version of the project uploaded to the repository.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Shreyas26X/SmartSense.git
cd SmartSense
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the Flask application:

```bash
python app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://127.0.0.1:5000/
```

---

## Machine Learning Model

SmartSense uses **K-Means clustering**, an unsupervised machine-learning algorithm.

The model groups customers according to behavioural similarity based primarily on:

- Recency
- Average Order Value

The project uses:

```text
K = 4
```

The resulting clusters are mapped into business-oriented customer segments.

---

## Evaluation

Since SmartSense uses **unsupervised clustering**, conventional classification metrics such as Accuracy, Precision, Recall and F1-score are not directly applicable unless ground-truth customer labels are available.

Suitable evaluation methods for the clustering system include:

- Inertia
- Silhouette Score
- Cluster separation
- Cluster interpretability

The current project focuses on generating meaningful and actionable customer segments.

---

## Application Screenshots

### Home / CSV Upload

Add a screenshot of the application's CSV upload interface here.

### Customer Segmentation Dashboard

Add a screenshot of the segmentation visualization here.

### Customer Segment Results

Add a screenshot of the generated customer segments here.

---

## Project Team

This project was developed collaboratively.

| Name | Role |
|---|---|
| Shreyas Indap | Project Member |
| Harsh Kanchan | Project Member |

---

## Academic Information

**Institution:** Atharva College of Engineering

**Department:** Artificial Intelligence & Data Science

**Class:** AIDS 2

**Semester:** VII

**Academic Year:** 2026–27

**Subject:** AI & DS II / Recent Open Source Project Lab (ROSPL)

**Guide:** Prof. Tejal Ranch

---

## Future Scope

The project can be further improved by:

- Adding more customer behavioural features
- Implementing RFM analysis
- Automatically determining the optimal number of clusters
- Adding Silhouette Score and other clustering metrics
- Integrating real-time customer data
- Developing personalized recommendation models
- Adding customer lifetime value prediction
- Deploying the application on a cloud platform
- Adding authentication and role-based access
- Integrating advanced machine-learning techniques

---

## License

This project is an academic mini project developed for educational purposes.

---

## Repository

GitHub Repository:

https://github.com/Shreyas26X/SmartSense

---

## Acknowledgement

We would like to express our sincere gratitude to **Prof. Tejal Ranch** for guidance and support throughout the development of this project.

We also thank **Atharva College of Engineering** for providing the opportunity and resources to develop this project as part of the AI & DS II and Recent Open Source Project Lab (ROSPL).
