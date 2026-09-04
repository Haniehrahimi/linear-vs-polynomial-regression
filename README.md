# Linear Regression vs Polynomial Regression

A simple comparison of Linear Regression and Polynomial Regression using the California Housing dataset.

## 📌 Project Overview

In this project, Linear Regression and Polynomial Regression are trained and evaluated on the California Housing dataset.

The goal is to compare the performance of both models using common regression metrics.

## 📊 Dataset

The project uses the **California Housing Dataset** provided by Scikit-learn.

* Samples: 20,640
* Features: 8
* Target: `MedHouseVal`

### Features

* `MedInc`
* `HouseAge`
* `AveRooms`
* `AveBedrms`
* `Population`
* `AveOccup`
* `Latitude`
* `Longitude`

## 🤖 Models

### 1. Linear Regression

A linear regression model is trained using the original features.

### 2. Polynomial Regression

Polynomial features with `degree=2` are generated and then used with Linear Regression.

## 📈 Evaluation Metrics

The models are evaluated using:

* **MAE (Mean Absolute Error)**
* **MSE (Mean Squared Error)**
* **R² Score**

## 📋 Results

| Model                 |    MAE |    MSE |     R² |
| --------------------- | -----: | -----: | -----: |
| Linear Regression     | 0.5332 | 0.5559 | 0.5758 |
| Polynomial Regression | 0.5162 | 0.6045 | 0.5387 |

## 🔎 Conclusion

Based on the test results, Linear Regression achieved a higher R² score than Polynomial Regression on this dataset.

Although Polynomial Regression had a slightly lower MAE, its R² score was lower than Linear Regression.

This shows that a more complex model does not always perform better on unseen data.

## 🛠️ Technologies

* Python
* Pandas
* Matplotlib
* Scikit-learn

## ▶️ How to Run

Clone the repository and install the required libraries:

```bash
pip install pandas matplotlib scikit-learn
```

Then run the Python script.

## 📁 Project Structure

```text
linear-vs-polynomial-regression/
│
├── main.py
├── README.md
└── .gitignore
```
<img width="640" height="480" alt="linear vs poly" src="https://github.com/user-attachments/assets/78ec9158-51b5-4c24-b990-386d29ac1b54" />
