from pyAudioAnalysis import audioTrainTest as aT

aT.featureAndTrain(["../Emotion Audio/01_Neatral", "../Emotion Audio/05_angry"], 1.0,1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "model_name", False)
aT.fileClassification("../Emotion Audio/06_Fear/03-01-06-01-01-01-01.wav", "model_name","svm")


# Result:(0.0, array([ 0.90156761,  0.09843239]), ['music', 'speech'])