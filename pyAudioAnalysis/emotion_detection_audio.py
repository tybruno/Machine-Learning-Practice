from pyAudioAnalysis import audioTrainTest as aT

aT.featureAndTrain(["../Emotion Audio/01_Neatral","../Emotion Audio/02_Calm","../Emotion Audio/03_Happy","../Emotion Audio/04_sad", "../Emotion Audio/05_angry","../Emotion Audio/06_Fear", "../Emotion Audio/07_disgust","../Emotion Audio/08_surprised"], 1.0,1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "emotion", False)


print(aT.fileClassification("../Emotion Audio/Testing data/03-01-08-02-02-02-01.wav", "emotion","svm"))
