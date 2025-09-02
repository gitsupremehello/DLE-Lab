import tensorflow as tf
from tensorflow.keras import layers, models
from glob import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# ---- Load and Preprocess Images ----
def load_img(files):
    return np.array([np.array(Image.open(f).convert('RGB').resize((64,64))) for f in files])

mel_files = glob('E:/Xai_Req_Setup/.../melanoma/*/*.*')
notmel_files = glob('E:/Xai_Req_Setup/.../notmelanoma/*/*.*')

X = np.concatenate([load_img(mel_files), load_img(notmel_files)])
y = np.concatenate([np.ones(len(mel_files)), np.zeros(len(notmel_files))])

X_train, X_test, y_train, y_test = train_test_split(X/255.0, y, test_size=0.2, random_state=42)

# ---- DNN Model ----
X_train_flat, X_test_flat = X_train.reshape(len(X_train), -1), X_test.reshape(len(X_test), -1)

dnn_model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(64*64*3,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

dnn_model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), 
                           tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

print("Training DNN...")
dnn_model.fit(X_train_flat, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
dnn_results = dnn_model.evaluate(X_test_flat, y_test, verbose=0)

# ---- CNN Model ----
cnn_model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), 
                           tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])

print("\nTraining CNN...")
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
cnn_results = cnn_model.evaluate(X_test, y_test, verbose=0)

# ---- Comparison ----
print("\n===== Model Comparison =====")
print(f"DNN - Loss: {dnn_results[0]:.4f}, Acc: {dnn_results[1]*100:.2f}%, "
      f"Prec: {dnn_results[2]:.4f}, Rec: {dnn_results[3]:.4f}, AUC: {dnn_results[4]:.4f}")

print(f"CNN - Loss: {cnn_results[0]:.4f}, Acc: {cnn_results[1]*100:.2f}%, "
      f"Prec: {cnn_results[2]:.4f}, Rec: {cnn_results[3]:.4f}, AUC: {cnn_results[4]:.4f}")
