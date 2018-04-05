from __future__ import division, print_function, absolute_import
import tflearn
import speech_data
import tensorflow as tf


# hdf5 is not supported on this machine (please install/reinstall h5py for optimal experience)
# curses is not supported on this machine (please install/reinstall curses for an optimal experience)

#learning rate. The higher the learning rate the faster the network trains.
# the slower the learning rate, the slower the network is trained but it is more accurate
learning_rate = 0.0001
training_iters = 300000  # steps we want to train for
batch_size = 64

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits (The number of digits that we are training 1- 9)

batch = word_batch = speech_data.mfcc_batch_generator(batch_size) #downloads the .wav files that have a recording of different spoken numbers
X, Y = next(batch) #labeled speach files
trainX, trainY = X, Y
testX, testY = X, Y #overfit for now

# uses Recurrent Neural Network (RNN)
#tensor is a multi dimensional array of data
# Network building

#width is the number of features abstracted from the utterances from our speech helper class
#height is the max length of each utterance
net = tflearn.input_data([None, width, height])

#128 is the number of neurons
#dropout helps prevents overfitting by turning of neurons during training.  This allows for a more generalized model
net = tflearn.lstm(net, 128, dropout=0.8) #LSTM is a network that remembers everything it has learned. USed for state of the art speech recognition
#softmax converts the numerical data into numerical probabilities
net = tflearn.fully_connected(net, classes, activation='softmax')
#regression will output a single predictive number for our utterance
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
# Training

### add this "fix" for tensorflow version errors
col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for x in col:
    tf.add_to_collection(tf.GraphKeys.VARIABLES, x ) 


model = tflearn.DNN(net, tensorboard_verbose=0)
while 1: #training_iters
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)
  _y=model.predict(X)
model.save("tflearn.lstm.model")
print (_y)
print (y)
