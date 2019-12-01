from keras.models import model_from_json
import numpy
import os
import tensorflow as tf

# Downloading the MNIST Dataset:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# import matplotlib.pyplot as plt
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
# plt.imshow(x_train[image_index], cmap='Greys')

x_train.shape

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

_ = input("Press any key to proceed for Training ... !")
model.fit(x=x_train,y=y_train, epochs=10)
print("MNIST Model Training Complete !!!")

print("Serializing Model to JSON file along with HD5F Weights file !!!")
# Serialize model to JSON
model_json = model.to_json()
with open("mnist_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("mnist_weights.h5")
print("Saved model to Disk")


_ = input("Press any key to proceed for model Evaluation  ... !")
modelEval = model.evaluate(x_test, y_test)
print("MNIST Model Evaluation Complete !!!")
print(modelEval)

img_rows = 28
img_cols = 28
image_index = 4444

_ = input("Press any key to proceed for  Inferencing/Predictions ... !")
true_pred = 0
false_pred = 0

for image in range(0, len(x_test)):
    pred_npy = model.predict(x_test[image].reshape(1, img_rows, img_cols, 1))
    pred = pred_npy.argmax()
    
    #print("Orinal Image: ", y_test[image])
    #print("Predicted: ", pred)
    if pred == y_test[image]:
        true_pred += 1
    if pred != y_test[image]:
        false_pred += 1

print("Total True Predictions on Test Dataset: ", true_pred)
print("Total False Predictions on Test Dataset: ", false_pred)

#print("Prediction for Input Image: ", y_test[image_index])
#print("Numpy Prediction Output for the Input Sample: ", pred)
#print(pred.argmax())

# ----------------------------------------- I M P O R T A N T ----------------------------------------- 
# Saving Keras model to Tensorflow Protocol Buffers .pb file format:
print('\n\n\n')
print ('Saving Keras Model to Tensorflow Protocol Buffers .pb file format ... ')
def freeze_session (session, keep_var_names=None, output_names=None, clear_devices=True):
    
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        
        return frozen_graph

from keras import backend as K
frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])

wkdir = './proto_files'
pb_filename = 'mnist_proto.pb'

tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)
print("SUCCESS: Model saved as Tensorflow Protocol Buffers .pb file to the speicified directory location !")