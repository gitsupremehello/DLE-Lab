import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from glob import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# --- Load Dataset ---
def load_img(files): 
    return np.array([np.array(Image.open(f).convert('RGB').resize((64,64))) for f in files])

X = np.concatenate([
    load_img(glob('E:/Xai_Req_Setup/Yashwanth 126156184/Skin_Image_Dataset/*/Skin*/skin_data/melanoma/*/*.*')),
    load_img(glob('E:/Xai_Req_Setup/Yashwanth 126156184/Skin_Image_Dataset/*/Skin*/skin_data/notmelanoma/*/*.*'))
])
y = np.concatenate([np.ones(len(X)//2), np.zeros(len(X)//2)])
X_train, X_test, y_train, y_test = train_test_split(X/255.0, y, test_size=0.2, random_state=42)

# --- Model ---
def create_model(opt):
    model = models.Sequential([
        layers.Conv2D(16, 3, activation='relu', input_shape=(64,64,3)), layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation='relu'), layers.MaxPooling2D(2),
        layers.Flatten(), layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Train Models ---
optimizers_dict = {"SGD": optimizers.SGD(0.01), "Adam": "adam"}
histories, models_dict = {}, {}

for name, opt in optimizers_dict.items():
    model = create_model(opt)
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    models_dict[name], histories[name] = model, history

# --- Plot Training & Validation Accuracy ---
plt.figure(figsize=(12,5))
for i, key in enumerate(["accuracy","val_accuracy"],1):
    plt.subplot(1,2,i)
    for name,hist in histories.items(): plt.plot(hist.history[key], label=f"{name} {key.title()}")
    plt.title(f'{key.title()} Comparison'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid()
plt.tight_layout(); plt.show()

# --- Evaluate Models ---
for name, model in models_dict.items():
    print(f"\n{name} Evaluation:")
    acc = model.evaluate(X_test, y_test, verbose=0)[1]*100
    print(f"Test Accuracy: {acc:.2f}%")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, y_pred))
    plt.plot(histories[name].history['accuracy'], label='Train')
    plt.plot(histories[name].history['val_accuracy'], label='Val')
    plt.title(f'{name} - Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(); plt.show()
    model.summary()
