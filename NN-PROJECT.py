import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Keras is an API designed for human beings, not machines.
# Keras is the most used deep learning framework among top-5 winning teams on Kaggle
# Keras follows best practices for reducing cognitive load:
# It offers consistent & simple APIs,
# It also has extensive documentation and developer guides.

# first step we will call mnist - database of handwritten digits available on - ... >
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# keras.utils.normalize() function calls the numpy.linalg.norm() to compute the norm and then
# normalize the input data. The given axis argument is
# therefore passed to norm() function to compute the norm along the given axis.
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# The sequential API allows you to create models layer-by-layer for most problems.
model = tf.keras.models.Sequential()
# A Flatten layer in Keras reshapes the tensor to have a shape
# that is equal to the number of elements contained in the tensor.
# This is the same thing as making a 1d-array of elements.
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# to make connections between the layers:
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
# to increase probability of correct answer:
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# attributes of my model like accuracy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.save('h_written.model')

model = tf.keras.models.load_model('h_written.model')



img_number = 1
while os.path.isfile(f"digits/{img_number}.png"):
    try:
        img = cv2.imread(f"digits/{img_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digits is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        img_number += 1
loss, accuracy = model.evaluate(x_test, y_test)
print("loss ratio is: ")
print(loss)
print("accuracy ratio is: ")
print(accuracy)
print("")
print("Thanks ... :)")
