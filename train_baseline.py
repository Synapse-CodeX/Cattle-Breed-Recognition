import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

DATA_DIR = "/kaggle/input/cattle-breed-recognition/dataset"
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Dataset not found at {DATA_DIR}. Attach the Kaggle dataset before running.")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
BASE_EPOCHS = 8
FINE_TUNE_EPOCHS = 6
SAVED_MODEL_DIR = "saved_model/breed_classifier"
os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    label_mode="int",
    shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    label_mode="int",
    shuffle=False
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
with open(os.path.join(SAVED_MODEL_DIR, "classes.json"), "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.12),
    layers.RandomTranslation(0.08, 0.08),
], name="data_augmentation")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_path = os.path.join(SAVED_MODEL_DIR, "best_model.h5")
cp = callbacks.ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1)
es = callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=BASE_EPOCHS,
    callbacks=[cp, es, reduce_lr]
)

base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.75)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

ft_checkpoint = os.path.join(SAVED_MODEL_DIR, "best_model_finetuned.h5")
ft_cp = callbacks.ModelCheckpoint(ft_checkpoint, monitor="val_accuracy", save_best_only=True, verbose=1)
ft_es = callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1)
ft_reduce = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[ft_cp, ft_es, ft_reduce]
)

model.save(SAVED_MODEL_DIR)


