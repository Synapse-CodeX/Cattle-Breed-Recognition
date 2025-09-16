import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_DIR = "/kaggle/input/cattle-breed-recognition/dataset"
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset not found at {DATA_DIR}. Attach the Kaggle dataset before running eval.")

MODEL_DIR = "saved_model/breed_classifier"
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Saved model not found at {MODEL_DIR}. Run training first.")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

model = tf.keras.models.load_model(MODEL_DIR)
with open(os.path.join(MODEL_DIR, "classes.json"), "r") as f:
    class_names = json.load(f)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

y_true = []
y_pred = []
for batch_imgs, batch_labels in test_ds:
    preds = model.predict(batch_imgs)
    y_pred.extend(np.argmax(preds, axis=1).tolist())
    y_true.extend(batch_labels.numpy().tolist())

acc = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
cm = confusion_matrix(y_true, y_pred)

os.makedirs("eval_results", exist_ok=True)
with open("eval_results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write(report)

pd.DataFrame(cm, index=class_names, columns=class_names).to_csv("eval_results/confusion_matrix.csv")
pd.DataFrame(class_names, columns=["class_name"]).to_csv("eval_results/class_names.csv", index=False)

print("Accuracy:", acc)
print("Saved eval_results/")

