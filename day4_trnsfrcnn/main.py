from src.data_loader import load_data
from src.model import build_model
from src.train import train_model

train_dir = "data/cats_vs_dogs/train"
val_dir = "data/cats_vs_dogs/validation"

train_data = load_data(train_dir)
val_data = load_data(val_dir)

model = build_model()
train_model(model, train_data, val_data)

model.save("transfer_cnn_model.h5")
