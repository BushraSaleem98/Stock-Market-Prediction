# Stock Market Prediction using LSTM

A deep learning project that predicts future stock prices using historical data and Stacked LSTM neural networks.

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Project Overview

This project implements a Stacked LSTM (Long Short-Term Memory) neural network to predict Apple (AAPL) stock prices based on historical data. The model learns patterns from past stock performance to forecast future prices.

## 🚀 Quick Start

### Installation

1. **Clone the repository**

   git clone https://github.com/yourusername/stock-market-prediction.git
   cd stock-market-prediction

2. Install dependencies

   pip install pandas numpy matplotlib scikit-learn tensorflow

3. Add your dataset

   Place AAPL.csv in the project directory

# Model Architecture
   Stacked LSTM Network:

   3 LSTM layers with 50 units each
   Dropout regularization (20%)
   Dense output layer for price prediction
   Adam optimizer with MSE loss function

#Training Parameters:
   Time steps: 100 (uses 100 previous days)
   Training split: 65% of data
   Test split: 35% of data
   Epochs: 100
   Batch size: 64

# Features
   Data Preprocessing: Automatic scaling and sequence generation
   Model Training: Stacked LSTM with validation monitoring
   Visualization: Training history and prediction plots
   Evaluation: RMSE and MAE metrics in dollars and percentages
   Professional Output: Clear progress tracking and results

# Output Metrics
   The model provides comprehensive evaluation:
   Root Mean Square Error (RMSE)
   Mean Absolute Error (MAE)
   Percentage errors relative to average price
   Training vs validation performance
   Prediction visualization charts
