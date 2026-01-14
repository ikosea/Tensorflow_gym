import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt 
import numpy as np 

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print("Training data shape: ", x_train.shape)
print("Test data shape: ", x_test.shape)


mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train_norm = (x_train - mean) / std
x_test_norm = (x_test - mean) / std


mdel = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(64, activation='relu'),  
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

history = model.fit(
    x_train_norm, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)

loss, mae = model.evaluate(x_test_norm, y_test, verbose=0)
print(f"Test Mean Squared Error: {loss:.2f}")
print(f"Test Mean Absolute Error: {mae:.2f}")


y_pred = model.predict(x_test_norm)

plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Boston Housing Price Prediction")
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training History')
plt.legend()
plt.show()
