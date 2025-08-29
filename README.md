# K-MEANS-AND-PRINCIPAL-COMPONENT-ANALYSIS-BASED-DATA-CLUSTERING-ON-TITANIC-DATASET
Implemented K-Means clustering and Principal Component Analysis (PCA) from scratch on the Titanic dataset using Age and Fare. Includes data preprocessing, dimensionality reduction, clustering, and visualizations. Outputs cluster summary, scatter plots, PCA projection, and reports in CSV, PDF, and DOCX.
# 🚢 Titanic Data Clustering – K-Means & PCA  

This project demonstrates unsupervised learning by implementing **K-Means Clustering** and **Principal Component Analysis (PCA)** **from scratch** in Python. The Titanic dataset is used to cluster passengers based on **Age** and **Fare**, helping uncover hidden patterns and socio-economic groups.  

---

## 📌 Project Overview  
- Implemented **K-Means** (no scikit-learn)  
- Implemented **PCA** (covariance, eigenvalues, eigenvectors)  
- Preprocessing: missing value handling (median imputation) & feature standardization  
- Visualizations:  
  - Scatter plots (original & standardized feature space)  
  - PCA projection (PC1 vs PC2)  
  - Explained variance chart  
- Generated formatted **reports (PDF & DOCX)** with tables & figures  

---

## 📊 Results  
The clustering identified **3 main groups** of passengers:  
1. **Low Fare, Young Passengers**  
2. **Moderate Fare, Middle-aged Passengers**  
3. **High Fare, Mixed Ages**  

PCA showed that **Fare** contributed most to overall variance.  

Cluster summary:  

| Cluster | Centroid Age | Centroid Fare | Size |  
|---------|--------------|---------------|------|  
| 0       | ~32.2        | ~211.5        | 40   |  
| 1       | ~35.9        | ~24.6         | 545  |  
| 2       | ~17.3        | ~22.3         | 306  |  

---

## 🛠️ Tech Stack  
- Python 3  
- NumPy, Pandas, Matplotlib  
- python-docx (for Word report)  

---

## 🚀 How to Run  
1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/titanic-kmeans-pca.git
   cd titanic-kmeans-pca
