import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import cv2

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Loaded model from disk")

img = cv2.imread('four.jpg',0)
img = img.reshape(1,1,28,28)
pred = loaded_model.predict_classes(img)
prob = loaded_model.predict_proba(img)
prob = "%.2f%%" % (prob[0][pred]*100)
print(pred[0],"is predicted with probablity",prob)
