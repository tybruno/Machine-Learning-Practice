from scipy.io.wavfile import _read_riff_chunk
from os.path import getsize

# filename = "..\\Emotion Audio\\01_Neatral\\03-01-01-01-01-01-01.wav"
filename = "C:\\Users\\Tyler\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\pyAudioAnalysis\\data\\speechEmotion\\00.wav"
with open(filename, 'rb') as f:
    riff_size, _ = _read_riff_chunk(f)

print('RIFF size: {}'.format(riff_size))
print('os size:   {}'.format(getsize(filename)))