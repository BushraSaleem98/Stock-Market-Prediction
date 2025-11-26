"""
Stock Market Prediction and Forecasting Using Stacked LSTM
This script loads historical Apple stock data, preprocesses it, builds an LSTM model,
trains it on the data, and evaluates its performance for stock price prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math
import os

def load_and_prepare_data():
    """
    Load the Apple stock data from CSV file and prepare it for processing.
    
    Returns:
        tuple: DataFrame and closing price series
    """
    # Correct file path for the AAPL.csv file
    file_path = r"C:\Users\lenovo\PycharmProjects\Stock-market-forcasting\AAPL.csv"
    
    try:
        # Load the dataset
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print("✓ Dataset loaded successfully")
        print(f"✓ Dataset shape: {df.shape}")
        
        # Display basic info about the dataset
        print(f"✓ Columns available: {df.columns.tolist()}")
        print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"✓ Number of trading days: {len(df)}")
        
        return df, df['close']
        
    except FileNotFoundError:
        print(f"✗ Error: File not found at {file_path}")
        print("Please make sure:")
        print("1. The file 'AAPL.csv' exists in the specified directory")
        print("2. The file path is correct")
        print("3. The file has read permissions")
        return None, None
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None, None

def explore_data(df, closing_prices):
    """
    Explore and visualize the initial data.
    
    Args:
        df (pd.DataFrame): Stock data DataFrame
        closing_prices (pd.Series): Closing price series
    """
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    # Display basic statistics
    print("\nBasic Statistics for Closing Prices:")
    print(f"Mean: ${closing_prices.mean():.2f}")
    print(f"Median: ${closing_prices.median():.2f}")
    print(f"Standard Deviation: ${closing_prices.std():.2f}")
    print(f"Minimum: ${closing_prices.min():.2f}")
    print(f"Maximum: ${closing_prices.max():.2f}")
    
    # Plot original closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(closing_prices, color='blue', linewidth=1)
    plt.title('Apple (AAPL) Historical Closing Prices', fontsize=14, fontweight='bold')
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Closing Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def preprocess_data(price_series):
    """
    Preprocess the data by scaling it to range [0,1] and splitting into train/test sets.
    
    Args:
        price_series (pd.Series): Series containing closing prices
    
    Returns:
        tuple: Scaled data, training data, test data, and scaler object
    """
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Reshape data for scaling (convert to 2D array)
    data = price_series.values.reshape(-1, 1)
    print(f"Original data shape: {data.shape}")
    
    # Initialize MinMax scaler and scale data to range [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    print("✓ Data scaled to range [0, 1]")
    
    # Split data into training (65%) and testing (35%)
    training_size = int(len(scaled_data) * 0.65)
    test_size = len(scaled_data) - training_size
    
    train_data = scaled_data[0:training_size]
    test_data = scaled_data[training_size:len(scaled_data)]
    
    print(f"✓ Training data size: {len(train_data)} samples ({training_size/len(scaled_data)*100:.1f}%)")
    print(f"✓ Test data size: {len(test_data)} samples ({test_size/len(scaled_data)*100:.1f}%)")
    
    return scaled_data, train_data, test_data, scaler

def create_sequences(dataset, time_step=100):
    """
    Create sequences of data for time series prediction.
    Each sequence contains 'time_step' previous values to predict the next value.
    
    Args:
        dataset (np.array): Scaled dataset
        time_step (int): Number of previous time steps to use for prediction
    
    Returns:
        tuple: Feature sequences (X) and target values (y)
    """
    dataX, dataY = [], []
    
    # Create sequences where each X contains 'time_step' previous values
    # and each Y contains the next value
    for i in range(len(dataset) - time_step - 1):
        sequence = dataset[i:(i + time_step), 0]  # Get 'time_step' previous values
        target = dataset[i + time_step, 0]        # Get the next value as target
        dataX.append(sequence)
        dataY.append(target)
    
    return np.array(dataX), np.array(dataY)

def build_lstm_model(input_shape):
    """
    Build and compile a Stacked LSTM model for time series prediction.
    
    Args:
        input_shape (tuple): Shape of input data (time_steps, features)
    
    Returns:
        Sequential: Compiled LSTM model
    """
    print("\n" + "="*50)
    print("BUILDING LSTM MODEL")
    print("="*50)
    
    # Create Sequential model
    model = Sequential()
    
    # First LSTM layer - return sequences for the next LSTM layer
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape, 
                  dropout=0.2, recurrent_dropout=0.2))
    
    # Second LSTM layer - return sequences
    model.add(LSTM(50, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    
    # Third LSTM layer - don't return sequences (last LSTM layer)
    model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
    
    # Output layer - single neuron for price prediction
    model.add(Dense(1))
    
    # Compile the model with appropriate loss function and optimizer
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    
    print("✓ LSTM model built successfully")
    print("✓ Model architecture:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train the LSTM model on the training data.
    
    Args:
        model: Compiled LSTM model
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
    
    Returns:
        History: Training history object
    """
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    print("Starting model training...")
    print(f"Training on {len(X_train)} sequences")
    print(f"Validating on {len(X_test)} sequences")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        verbose=1,
        shuffle=False  # Don't shuffle time series data
    )
    
    print("✓ Model training completed")
    return history

def plot_training_history(history):
    """
    Plot the training history to visualize model learning.
    
    Args:
        history: Training history object from model.fit()
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training & validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    """
    Evaluate the model and make predictions.
    
    Args:
        model: Trained LSTM model
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        scaler: Fitted scaler object
    
    Returns:
        tuple: Training predictions, test predictions, and performance metrics
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    print("Making predictions...")
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Transform predictions back to original price scale
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    
    # Transform actual values back to original scale for evaluation
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate RMSE (Root Mean Square Error)
    train_rmse = math.sqrt(mean_squared_error(y_train_actual, train_predict))
    test_rmse = math.sqrt(mean_squared_error(y_test_actual, test_predict))
    
    # Calculate additional metrics
    train_mae = np.mean(np.abs(y_train_actual - train_predict))
    test_mae = np.mean(np.abs(y_test_actual - test_predict))
    
    return train_predict, test_predict, train_rmse, test_rmse, train_mae, test_mae

def plot_predictions(original_data, train_predict, test_predict, look_back, scaler):
    """
    Plot the original data along with training and test predictions.
    
    Args:
        original_data (np.array): Original scaled data
        train_predict (np.array): Training predictions
        test_predict (np.array): Test predictions
        look_back (int): Number of time steps used for prediction
        scaler: Fitted scaler object
    """
    # Prepare training predictions for plotting
    train_predict_plot = np.empty_like(original_data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict
    
    # Prepare test predictions for plotting
    test_predict_plot = np.empty_like(original_data)
    test_predict_plot[:, :] = np.nan
    test_start = len(train_predict) + (look_back * 2) + 1
    test_predict_plot[test_start:len(original_data) - 1, :] = test_predict
    
    # Convert back to original scale for plotting
    original_data_unscaled = scaler.inverse_transform(original_data)
    
    # Create the main prediction plot
    plt.figure(figsize=(14, 8))
    
    plt.plot(original_data_unscaled, label='Actual Prices', color='blue', linewidth=1.5, alpha=0.7)
    plt.plot(train_predict_plot, label='Training Predictions', color='green', linewidth=1.5, alpha=0.8)
    plt.plot(test_predict_plot, label='Test Predictions', color='red', linewidth=1.5, alpha=0.8)
    
    plt.title('Apple Stock Price Prediction using LSTM', fontsize=16, fontweight='bold')
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """Main function to execute the complete stock price prediction pipeline."""
    
    print("🚀 Starting Stock Price Prediction using LSTM")
    print("=" * 60)
    
    # Step 1: Load and explore data
    df, closing_prices = load_and_prepare_data()
    if df is None:
        print("❌ Failed to load data. Exiting program.")
        return
    
    # Explore the loaded data
    explore_data(df, closing_prices)
    
    # Step 2: Preprocess the data
    scaled_data, train_data, test_data, scaler = preprocess_data(closing_prices)
    
    # Step 3: Create sequences for time series prediction
    time_step = 100  # Use 100 previous days to predict the next day
    
    print(f"\nCreating sequences with {time_step} time steps...")
    X_train, y_train = create_sequences(train_data, time_step)
    X_test, y_test = create_sequences(test_data, time_step)
    
    print(f"✓ Training sequences: {X_train.shape}")
    print(f"✓ Training targets: {y_train.shape}")
    print(f"✓ Test sequences: {X_test.shape}")
    print(f"✓ Test targets: {y_test.shape}")
    
    # Reshape data for LSTM input [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Step 4: Build LSTM model
    model = build_lstm_model((time_step, 1))
    
    # Step 5: Train the model
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Step 6: Evaluate the model
    train_predict, test_predict, train_rmse, test_rmse, train_mae, test_mae = evaluate_model(
        model, X_train, y_train, X_test, y_test, scaler
    )
    
    # Step 7: Display results
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    # Calculate percentage errors for better interpretation
    avg_price = np.mean(closing_prices)
    test_rmse_percentage = (test_rmse / avg_price) * 100
    test_mae_percentage = (test_mae / avg_price) * 100
    
    print(f"Training RMSE: ${train_rmse:.2f}")
    print(f"Test RMSE:     ${test_rmse:.2f} ({test_rmse_percentage:.2f}% of average price)")
    print(f"Training MAE:  ${train_mae:.2f}")
    print(f"Test MAE:      ${test_mae:.2f} ({test_mae_percentage:.2f}% of average price)")
    
    print(f"\nAverage Stock Price: ${avg_price:.2f}")
    
    # Step 8: Plot final predictions
    print("\nGenerating prediction visualization...")
    plot_predictions(scaled_data, train_predict, test_predict, time_step, scaler)
    
    print("\n" + "🎯 STOCK PRICE PREDICTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

# Run the main function
if __name__ == "__main__":
    main()
