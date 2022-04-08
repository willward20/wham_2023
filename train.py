import tensorflow as tf
from tensorflow.keras import layers, models


model = models.Sequential(
    [
        tf.keras.Input(shape=(150, 150, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(24, 5, 2, activation="relu"),
        layers.Dropout(0.2),
        layers.Conv2D(32, 5, 2, activation="relu"),
        layers.Dropout(0.2),
        layers.Conv2D(64, 5, 2, activation="relu"),
        layers.Dropout(0.2),
        layers.Conv2D(64, 3, 2, activation="relu"),
        layers.Dropout(0.2),
        layers.Conv2D(64, 3, 1, activation="relu"),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(50, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(3, activation="softmax"),
    ]
)

# verify model
model.summary()

# load data
# create training dataset
data_dir = "./data_04081025/"
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(150, 150),
    batch_size=32,
)
# create validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(150, 150),
    batch_size=32,
)
# AUTOTUNE = tf.data.AUTOTUNE
# train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# train
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(train_ds, validation_data=val_ds, epochs=100)

# save model
model.save(f'model_{data_dir.split("_")[-1]}')
