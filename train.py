from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.preprocessing.image import img_to_array
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator



train_images = []
train_labels = []

directory = 'dataset'
label_encoder_path = "label_encoder.pkl"

for root, dirs, files in os.walk(directory):
    for file in files:
        file_name= os.path.join(root, file)
        img = cv2.imread(file_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_normalized = img/255
        resized_img = cv2.resize(img_normalized, (256, 256))
        image = img_to_array(resized_img)
        train_images.append(image)    

        # Extract the label from the directory name
        label = os.path.basename(root)
        train_labels.append(label)
        

train_images = np.array(train_images) 
train_labels = np.array(train_labels) 
print(train_images.shape,train_labels.shape)


label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
print(train_labels)
train_labels = tf.keras.utils.to_categorical(train_labels,3)
with open(label_encoder_path, 'wb') as file:
    pickle.dump(label_encoder, file)

train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=20,
   
)

# Generate augmented images and labels
augmented_train_generator = datagen.flow(
    train_images, train_labels, batch_size=32, shuffle=True
)

model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu',strides=(1, 1), input_shape=(256,256, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu',strides=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist=model.fit(augmented_train_generator, epochs=20,validation_data=(test_images, test_labels))
model.save('game_model.h5')

# Plot training and validation accuracy
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()