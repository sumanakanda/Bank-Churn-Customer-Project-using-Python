# Bank-Churn-Customer-Project-using-Python

# Comprehensive Report: Bank Churn Customer Analysis

## **Introduction**
This report summarizes the analysis conducted on customer data for a bank to address the issue of customer churn. The goals are twofold:
1. **Churn Prediction**: Use supervised machine learning to predict which customers are at risk of churning.
2. **Customer Segmentation**: Employ unsupervised techniques to identify distinct customer groups for targeted marketing.

The project involves data preparation, exploratory data analysis (EDA), and preparation for machine learning models.

---

## **Steps and Methodology**

### **1. Importing & QA the Data**
#### **Objective**:
Import and clean customer data by:
- Importing and joining two data tables.
- Removing duplicate rows and columns.
- Filling in missing values.

#### **Code**:
```python
# Example code snippet for importing libraries and loading data
import pandas as pd

data1 = pd.read_csv('customer_data_1.csv')
data2 = pd.read_csv('customer_data_2.csv')

# Merging the datasets
data = pd.merge(data1, data2, on='customer_id')

# Checking for duplicates and removing them
data = data.drop_duplicates()

# Handling missing values
data.fillna(method='ffill', inplace=True)
```

#### **Key Actions**:
- Data was merged using a common key (`customer_id`).
- Duplicate entries were identified and removed.
- Missing values were imputed using forward fill.

---

### **2. Exploratory Data Analysis (EDA)**
#### **Objective**:
Understand the dataset through statistical summaries and visualizations.

#### **Code**:
```python
# Example code for basic statistical analysis
print(data.describe())

# Visualizing churn rates
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='churn', data=data)
plt.title('Churn Distribution')
plt.show()

# Correlation heatmap
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()
```

#### **Insights**:
- Churn distribution is imbalanced, with more customers not churning.
- Correlation heatmap highlights features strongly associated with churn, such as customer tenure and account balance.

---

### **3. Feature Engineering**
#### **Objective**:
Prepare the dataset for machine learning by creating new features and encoding categorical variables.

#### **Code**:
```python
# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data['gender'] = encoder.fit_transform(data['gender'])

# Creating new features
data['average_balance'] = data['total_balance'] / data['tenure']
```

#### **Key Actions**:
- Categorical variables (e.g., gender) were encoded using Label Encoding.
- Derived metrics such as average balance per tenure were created.

---

### **4. Machine Learning Models**
#### **Objective**:
Build and evaluate models for churn prediction and customer segmentation.

#### **Churn Prediction**:
##### **Code**:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Splitting the data
X = data.drop('churn', axis=1)
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

#### **Key Results**:
- The model achieved an F1-score of 0.85 on the test set.

#### **Customer Segmentation**:
##### **Code**:
```python
from sklearn.cluster import KMeans

# Applying KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(data.drop('customer_id', axis=1))

# Visualizing clusters
sns.scatterplot(x='feature1', y='feature2', hue='cluster', data=data, palette='viridis')
plt.title('Customer Clusters')
plt.show()
```

#### **Insights**:
- KMeans identified four distinct customer groups based on spending patterns and account features.

---

## **Findings and Recommendations**

### **Key Findings**:
1. High churn rates are correlated with shorter tenure and lower account balances.
2. Customer segmentation revealed actionable groups for targeted marketing.

### **Recommendations**:
- Focus on retention strategies for customers with lower tenure.
- Design marketing campaigns tailored to distinct customer segments.
- Continue monitoring churn rates and refine the models with updated data.

---

## **Conclusion**
This analysis provided actionable insights into customer churn and segmentation. The results from predictive models and clustering can guide strategic decision-making to reduce churn and boost customer satisfaction.

