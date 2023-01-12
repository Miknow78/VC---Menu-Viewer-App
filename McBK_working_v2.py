from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from skimage.transform import resize
import numpy as np
import os

# Improting Image class from PIL module  
from PIL import Image  


# -------------------------------------------
# FUNCTIONS
# -------------------------------------------

def readandresize_images(path):
    # Resize images
    im1 = Image.open(path)  
    newsize = (IMAGE_SIZE, IMAGE_SIZE) 
    im1 = im1.resize(newsize) 
    im1 = np.asarray(im1)
    return im1

def load_images(image_paths):
    # Load the images from disk.
    images = [readandresize_images(path) for path in image_paths]
    return np.asarray(images)

def createlabels(n):
    t = []
    for i in range(n*ncat):
    	if i <= n-1: t = t + [0]
    	elif i <= 2*n-1: t = t + [1]
    	elif i <= 3*n-1: t = t + [2]
    	elif i <= 4*n-1: t = t + [3]
    	elif i <= 5*n-1: t = t + [4]
    lbls = np.asarray(t)
    return lbls

def unison_shuffle(a, b):
    # ----shuffle data ----
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# -------------------------------------------
# VARIABLES
# -------------------------------------------
    
IMAGE_SIZE = 128

folder_main = "C:/Users/Utilizador/AndroidStudioProjects/MyApplication/Imagens/"
chains = ["Mc", "BK", "KFC", "SW", "PZ"]
ncat = len(chains)
ntrain = 40
ntest = 10 

## -----------------------------------
## Load data
## -----------------------------------

## -------- TRAIN ---------
img_paths =[]
for c in chains:
	curfolder = folder_main + c +"/"
	for i in range(ntrain):
    		img_paths = img_paths +  [curfolder + "image (" + str(i+1) + ").jpg"]
train_images = load_images(image_paths=img_paths)

## create train labels
train_labels = createlabels(ntrain)
len(train_labels)

## -------- TEST ---------
img_paths =[]
for c in chains:
	curfolder = folder_main + c +"/"
	for i in range(ntest):
    		img_paths = img_paths +  [curfolder + "image (" + str(i+1) + ").jpg"]
test_images = load_images(image_paths=img_paths)

test_labels = createlabels(ntest)
len(test_labels)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


# shuffle
train_images, train_labels = unison_shuffle(train_images, train_labels)
test_images, test_labels = unison_shuffle(test_images, test_labels)



plt.figure()
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(chains[test_labels[i]])
plt.show()


## -----------------------------------
## Modelo
## -----------------------------------
model = models.Sequential()
model.add(layers.Conv2D(IMAGE_SIZE, (5,5), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(36, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(24, (3, 3), activation='relu'))


model.add(layers.Flatten())
model.add(layers.Dense(36, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_split = 0.1)

print('\nhistory dict:', history.history)

plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)



y_prob = model.predict(test_images) 
y_classes_test = y_prob.argmax(axis=-1)
#confirm print
plt.figure()
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(chains[y_classes_test[i]])
plt.show()

# predict 

folder_gmaps= "C:/Users/Utilizador/AndroidStudioProjects/MyApplication/Imagens/teste/"

img_paths= []
for i in range(5):
    img_paths = img_paths +  [folder_gmaps + "ol (" + str(i+1) + ").jpeg"]

img_paths = img_paths +  [folder_gmaps + "ol (6).jpg"]
img_paths = img_paths +  [folder_gmaps + "ol (7).jpg"]

test_images_gmaps = load_images(image_paths=img_paths)


# Normalize pixel values to be between 0 and 1
test_images_gmaps = test_images_gmaps / 255.0


y_prob = model.predict(test_images_gmaps) 
y_classes = y_prob.argmax(axis=-1)
print("Nr of images wrong:")

#confirm print
plt.figure()
for i in range(7):
    plt.subplot(1,7,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    print(i)
    print(chains[y_classes[i]])
    plt.imshow(test_images_gmaps[i], cmap=plt.cm.binary)
    plt.xlabel(chains[y_classes[i]])
plt.show()










