import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import tensorflow as tf

model = models.load_model("btc_model.keras")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

img_path = './testing/glioma/Te-gl_0011.jpg'
img = cv.imread(img_path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32))

img = img / 255.0

plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()

img_array = np.expand_dims(img, axis=0)
prediction = model.predict(img_array)
index = np.argmax(prediction)
print(class_names[index])

# print(model.summary())