from pathlib import Path
import imghdr
import os

data_dir = "hair_types"
image_extensions = [".png", ".jpg"]  # add there all your images file extensions

# img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
# for filepath in Path(data_dir).rglob("*"):
#     if filepath.suffix.lower() in image_extensions:
#         img_type = imghdr.what(filepath)
#         if img_type is None:
#             print(f"{filepath} is not an image")
#             os.remove(filepath)
#         elif img_type not in img_type_accepted_by_tf:
#             print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
#             os.remove(filepath)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


image_size = (512, 256)
batch_size = 8

# train_ds = tf.keras.utils.image_dataset_from_directory(
#     "hair_types",
#     validation_split=0.3,
#     subset="training",
#     seed=1337, #original 1337
#     image_size=image_size,
#     batch_size=batch_size,
#     labels='inferred',
#     label_mode='categorical'
# )
# val_ds = tf.keras.utils.image_dataset_from_directory(
#     "hair_types",
#     validation_split=0.7,
#     subset="validation",
#     seed=1337,
#     image_size=image_size,
#     batch_size=batch_size, 
#     labels='inferred',
#     label_mode='categorical'
# )


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Input

# TRYING ALEXNET ARCHITECHTURE
# model = keras.models.Sequential([
#     Input(shape=(227,227,3)),
#     keras.layers.RandomRotation(0.2),
#     keras.layers.RandomBrightness(0.2),

#     keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),

#     keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),

#     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),

#     keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),

#     keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
#     keras.layers.BatchNormalization(),
    
#     keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.1),
#     keras.layers.Dense(4096, activation='relu'),
#     keras.layers.Dropout(0.1),
#     keras.layers.Dense(3, activation='softmax')
# ])
# model.summary()

# TRYING VGG 19 ARCHITECTURE
# model = Sequential([
#     Input(shape=(224,224,3)),
#     layers.Rescaling(1.0 / 255),

#     layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(pool_size=(2,2)),
#     layers.Dropout(.1),

#     layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(pool_size=(2,2)),
#     layers.Dropout(.1),

#     layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(pool_size=(2,2)),
#     layers.Dropout(.2),

#     layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(pool_size=(2,2)),
#     layers.Dropout(.2),

#     layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
#     layers.BatchNormalization(),
#     layers.MaxPooling2D(pool_size=(2,2)),
#     layers.Dropout(.3),

#     layers.Flatten(),
#     layers.Dense(4096, activation='relu'),
#     layers.Dropout(.5),
#     layers.Dense(4096, activation='relu'),
#     layers.Dropout(.5),
#     layers.Dense(3, activation='softmax')
# ])

model = Sequential()
model.add(keras.Input(shape=image_size + (1,)))

model.add(layers.Conv2D(filters=8, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(20))
model.add(layers.Dense(1, activation='sigmoid'))

# tf.keras.utils.plot_model(model, to_file='model_test.png', show_shapes=True)

print(model.summary())
epochs = 50

# model.compile(
#     optimizer=keras.optimizers.Adam(1e-3),
#     loss=tf.keras.losses.CategoricalCrossentropy(),
#     metrics=["accuracy"],
# )

# history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, shuffle=True)

# import matplotlib.pyplot as plt
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# print('average accuracy: ', np.mean(history.history['accuracy']))
# print('average val_accuracy: ', np.mean(history.history['val_accuracy']))

# images_dict = {
#     'Curly_Hair' : list(Path(data_dir).rglob("Curly_Hair/*")),
#     'Straight_Hair' : list(Path(data_dir).rglob("Straight_Hair/*")),
#     'Wavy_Hair' : list(Path(data_dir).rglob("Wavy_Hair/*"))
# }
# for key, value in images_dict.items():

#     print(f"\n\n===== NOW PREDICTING {key} =====\n\n")
#     rand_hair = np.random.choice(value,10,replace=False)

#     for i in rand_hair:
#         img = keras.preprocessing.image.load_img(i, target_size=image_size)
#         img_array = keras.preprocessing.image.img_to_array(img)
#         img_array = tf.expand_dims(img_array, 0)  # Create batch axis

#         predictions = model.predict(img_array)
#         print(
#             "image path: " + i.__str__() + "\n" +
#             "This image is %.2f percent curly hair, %.2f percent straight hair, and %.2f percent wavy hair.\n"
#             % tuple(predictions[0])
#         )