from tensorflow.python.platform import gfile
import tensorflow as tf
import time

# Downloading the MNIST Dataset:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
print('Training Set Shape: ', x_train.shape)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print('Testing Set Shape: ', y_test.shape)

print('Taining Set Shape: ', x_train.shape)

wkdir = './proto_files'
pb_filename = 'mnist_proto.pb'

with tf.Session() as sess:
    # Loading Model from Protocol Buffers .pb format file:
    
    with gfile.FastGFile(wkdir + '/' + pb_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        g_in = tf.import_graph_def(graph_def)
        
    # Write to Tensorboard (check tensorboard for each op names):
    writer = tf.summary.FileWriter(wkdir + '/log/')
    writer.add_graph(sess.graph)
    writer.flush()
    writer.close()
    
    print("SUCCESS: Tensorflow Protocol Buffers .pb File Loaded Successfully from the Disk !")

    # Inferencing using loaded Protocol Buffers .pb Model file:
    opr = sess.graph.get_operations()
    #for opr in sess.graph.get_operations():
        #if opr.name == 'import/dense_2/Softmax':
    #    print(opr.name)

    tensor_output = sess.graph.get_tensor_by_name('import/dense_2/Softmax:0')
    print("SUCCESS: Output tensor retreived successfully from the graph operations !")
    print(tensor_output)

    tensor_input = sess.graph.get_tensor_by_name("import/conv2d_1_input:0")
    print("SUCCESS: Input tensor retreived successfully from the graph operations !")
    print(tensor_input)
    #print(op[0])

    img_rows = 28
    img_cols = 28
    
    # ----------------------------------------- I M P O R T A N T -----------------------------------------
    # Inferencing over MNIST Dataset (Testset) using Protocol Buffers .pb Model file:
    
    # _ = input("Press any key to proceed for  Inferencing/Predictions ... !")
    print("\n\nProceeding for Inference on MNIST Dataset ... !")
    start_time = time.time()
    true_pred = 0
    false_pred = 0

    for image in range(0, len(x_test)):
        #pred_npy = model.predict(x_test[image].reshape(1, img_rows, img_cols, 1))
        #pred = pred_npy.argmax()
        
        prediction = (sess.run(tensor_output, {tensor_input: x_test[image].reshape(-1, img_rows, img_cols, 1)})).argmax()
        #print('Prediction: ', prediction)
        #print("Orinal Image: ", y_test[image])
        #print("Predicted: ", pred)
        if prediction == y_test[image]:
            true_pred += 1
        if prediction != y_test[image]:
            false_pred += 1

    print("Total True Predictions on Test Dataset: ", true_pred)
    print("Total False Predictions on Test Dataset: ", false_pred)
    end_time = time.time()
    
    print("Inferencing Time: 10,000 test images (secs): ", end_time - start_time)
    print("Inferencing Time: Images processed / sec: ", 10000/(end_time - start_time))