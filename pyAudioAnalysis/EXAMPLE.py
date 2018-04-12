from pyAudioAnalysis import audioTrainTest as aT

aT.featureAndTrain(["..\\Emotion Audio\\01_Neatral\\16bit", "..\\Emotion Audio\\05_angry"], 1.0,1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "emotion", False)
# aT.featureAndTrain(["test1", "test2"], 1,1, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)

print(aT.fileClassification("..\\Emotion Audio\\06_Fear\\03-01-06-01-01-01-01.wav", "emotion","svm"))

#
# array = [ 0.90156761,  0.09843239]
# Result:(0.0, array, ['neatral', 'angry'])

# from pyAudioAnalysis import audioTrainTest as aT
# aT.featureAndTrain(["classifierData/music","classifierData/speech"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmSMtemp", False)
# aT.fileClassification("data/doremi.wav", "svmSMtemp","svm")
