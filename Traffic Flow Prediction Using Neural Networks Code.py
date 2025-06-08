import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("traffic.csv")

# Convert the DateTime column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Set DateTime as the index
df.set_index('DateTime', inplace=True)

# List of junctions
junctions = [1, 2, 3, 4]

# Results dictionary
results = {}

# Iterate over each junction
for j in junctions:
    print(f"Processing Junction {j}...")
    
    # Filter data for the junction
    df_junction = df[df['Junction'] == j]['Vehicles']
    
    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_junction_scaled = scaler.fit_transform(df_junction.values.reshape(-1, 1))
    
    # Split data into training and testing sets
    train_size = int(len(df_junction_scaled) * 0.8)
    train, test = df_junction_scaled[:train_size], df_junction_scaled[train_size:]
    
    # Create dataset function
    def create_dataset(dataset, seq_length=1):
        X, y = [], []
        for i in range(len(dataset) - seq_length - 1):
            X.append(dataset[i:(i + seq_length), 0])
            y.append(dataset[i + seq_length, 0])
        return np.array(X), np.array(y)
    
    seq_length = 24
    X_train, y_train = create_dataset(train, seq_length)
    X_test, y_test = create_dataset(test, seq_length)
    
    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model with validation data
    history = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stop])
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train_actual = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test_actual = scaler.inverse_transform([y_test])
    
    # Performance evaluation
    train_mse = mean_squared_error(y_train_actual[0], train_predict)
    train_mae = mean_absolute_error(y_train_actual[0], train_predict)
    test_mse = mean_squared_error(y_test_actual[0], test_predict)
    test_mae = mean_absolute_error(y_test_actual[0], test_predict)

    # Save results
    results[j] = {
        'Train MSE': train_mse,
        'Train MAE': train_mae,
        'Test MSE': test_mse,
        'Test MAE': test_mae,
    }

    # Create a new dataframe to align the predictions with the dates
    train_predict_plot = np.empty_like(df_junction_scaled)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[seq_length:len(train_predict) + seq_length, :] = train_predict

    test_predict_plot = np.empty_like(df_junction_scaled)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (seq_length * 2) + 1:len(df_junction_scaled) - 1, :] = test_predict

    # Plot LSTM forecast
    plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(df_junction_scaled), label='Actual Data')
    plt.plot(train_predict_plot, label='LSTM Training Forecast')
    plt.plot(test_predict_plot, label='LSTM Test Forecast')
    plt.title(f'LSTM Forecast for Junction {j}')
    plt.xlabel('DateTime')
    plt.ylabel('Number of Vehicles')
    plt.legend()
    plt.show()

    # Plot the training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss for Junction {j}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Display results
results_df = pd.DataFrame.from_dict(results, orient='index')
# Display separate results for train and test
print("Train Results:")
for j in junctions:
    print(f"Junction {j}:")
    print(f"  Train MSE: {results[j]['Train MSE']}")
    print(f"  Train MAE: {results[j]['Train MAE']}")
    print()

print("Test Results:")
for j in junctions:
    print(f"Junction {j}:")
    print(f"  Test MSE: {results[j]['Test MSE']}")
    print(f"  Test MAE: {results[j]['Test MAE']}")
    print()

