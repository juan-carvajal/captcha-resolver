from tensorflow import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
model = keras.models.load_model('model.h5')

path = 'parts/1/1ASBM-1.png'

test_image = image.load_img(path, target_size=(30, 15))
plt.imshow(test_image)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)[0]
index = np.where(result == 1)
cats = list('123456789ABCDEFHKMNPRSTUVWXYZ')
print(cats[index[0][0]])




