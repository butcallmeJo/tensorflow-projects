# TensorFlow and to use tf.keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import random

print(tf.__version__)

"""
This file trains a neural network model to classify images of clothing,
like different shoes, tops and more.
"""

"""
Use the mnist fashion dataset, use to get np arrays of training and test data. 
The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255. 
The labels are an array of integers, ranging from 0 to 9. 
These represent the class of clothing that the image represents:
0 	T-shirt/top
1 	Trouser
2 	Pullover
3 	Dress
4 	Coat
5 	Sandal
6 	Shirt
7 	Sneaker
8 	Bag
9 	Ankle boot
"""
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(f"train_images shape: {train_images.shape}, train_labels len: {len(train_labels)}")
print(f"test_images shape: {test_images.shape}, test_labels len: {len(test_labels)}")

# TODO: use cmd line arg if you want to see below 
# quick look at first image from training_images using matplotlib.pyplot
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

"""
Preprocess data: pixel values are 0-255. NN needs 0-1
"""
train_images = train_images / 255.0
test_images = test_images / 255.0

# TODO: use cmd line arg if you want to see below 
# make sure training data is accurate by showing first 25
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary) # type: ignore -- throwing unkwn attr
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


"""
Building Model
- set up layers
- compile model
"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # convert 2d array to 1d
    tf.keras.layers.Dense(128, activation='relu'), # 128 nodes/neurons
    tf.keras.layers.Dense(10) # returns a logits array with length of 10 (class of clothing)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

"""
Training Model
- Feed/fit model
- Evaluate accuracy
- Get predictions
- TODO: address overfit
"""
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2) # type: ignore (verbose default is a str...but accepts ints)
print(f"\nTest accuracy: {test_acc}")

"""
Predictions
"""
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
# print(f"predictions[0]: {predictions[0]}")
# print(f"all predictions: {predictions}")

# test 0 prediction
if np.argmax(predictions[0]) == test_labels[0]:
    print("Got the correct prediction for test 0")
else:
    print("Got the wrong prediction for test 0")

"""
Functions to graph the full set of 10 class predictions.
use to verify predictions visually
"""
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary) # type: ignore -- throwing unkwn attr

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# i = 0
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()

# i = 12
# plt.figure(figsize=(6,3))
# plt.subplot(1,2,1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1,2,2)
# plot_value_array(i, predictions[i],  test_labels)
# plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
# TODO: use cmd line arg to show
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()

"""
Use the model on random img (article of clothing)
"""
print("\nUsing the model on a random image from the test dataset")
# Grab an image from the test dataset.
img_idx = random.randint(0, len(test_images) - 1)
img = test_images[img_idx]
print(f"img.shape 1: {img.shape}")
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)
# predict the correct label for this image
predictions_single = probability_model.predict(img)
print(predictions_single)
# show visually TODO: use cmd line arg to show?
    # plot_value_array(1, predictions_single[0], test_labels)
    # _ = plt.xticks(range(10), class_names, rotation=45)
    # plt.show()

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(img_idx, predictions_single[0], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(img_idx, predictions_single[0],  test_labels)
plt.show()

# test 0 prediction
print(f"Predicted label for test {img_idx}: {np.argmax(predictions_single[0])}")
print(f"Actual label for test {img_idx}: {test_labels[img_idx]}")
if np.argmax(predictions_single[0]) == test_labels[img_idx]:
    print(f"Got the correct prediction for test {img_idx}")
else:
    print(f"Got the wrong prediction for test {img_idx}")
