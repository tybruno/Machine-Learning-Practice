from scipy.signal import hanning
import tensorflow as tf
import numpy as np

N = 256 # FFT size
audio = np.random.rand(N, 1) * 2 - 1
w = hanning(N)

input  = tf.placeholder(tf.float32, shape=(N, 1))
window = tf.placeholder(tf.float32, shape=(N))
window_norm = tf.div(window, tf.reduce_sum(window))
windowed_input = tf.multiply(input, window_norm)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    windowed_input_val = sess.run(windowed_input, {
        window: w,
        input: audio
    })

N = 512 # FFT size
input_length = int(input.get_shape()[1])

zeros_left = tf.zeros([int(input.get_shape()[0]), int((N - input_length+1) / 2)])
zeros_right = tf.zeros([int(input.get_shape()[0]), int((N - input_length) / 2)])
input_padded = tf.concat([zeros_left, input, zeros_right], axis=1)

fftbuffer_left  = tf.slice(windowed_input, [0, int(N/2)], [-1, -1])
fftbuffer_right = tf.slice(windowed_input, [0, 0],   [-1, int(N/2)])
fftbuffer = tf.concat([fftbuffer_left, fftbuffer_right], axis=1)

fft = tf.fft(tf.cast(fftbuffer, tf.complex64))

# compute absolute value of positive side
sliced_fft = tf.slice(fft, [0, 0], [-1, positive_spectrum_size])
abs_fft = tf.abs(sliced_fft)

# magnitude spectrum of positive frequencies in dB
magnitude = 20 * log10(tf.maximum(abs_fft, 1E-06))

# phase of positive frequencies
phase = angle(sliced_fft)