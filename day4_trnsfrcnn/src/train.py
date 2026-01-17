from tensorflow.keras.callbacks import ModelCheckpoint

def train_model(model, train_data, val_data, epochs=10):
    checkpoint = ModelCheckpoint(
        "model_epoch_{epoch}.keras",
        save_best_only=False
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    return history
