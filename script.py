import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

(xtrain,ytrain),(xtest,ytest) = mnist.load_data()

xtrain = xtrain.reshape(xtrain.shape[0],1,28,28)
xtest = xtest.reshape(xtest.shape[0],1,28,28)
xtrain = xtrain/255
xtest = xtest/255

ytrain = np_utils.to_categorical(ytrain)
ytest = np_utils.to_categorical(ytest)

num_classes = ytest.shape[1]
print(num_classes)

def baseline_model():
        model = Sequential()
        model.add(Conv2D(30,(5,5),input_shape=(1,28,28),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(15,(3,3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128,activation='relu'))
        model.add(Dense(50,activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

model = baseline_model()

model.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=10,batch_size=200)
model.evaluate(xtest,ytest,verbose=0)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
scores = model.evaluate(xtest,ytest,verbose=0)
print("\nAccuracy : %.2f%%" % (scores[1]*100))



