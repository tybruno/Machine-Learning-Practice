from pyAudioAnalysis import audioTrainTest as aT
import threading
import os

aT.featureAndTrain(["../Emotion Audio/01_Neatral","../Emotion Audio/02_Calm","../Emotion Audio/03_Happy","../Emotion Audio/04_sad", "../Emotion Audio/05_angry","../Emotion Audio/06_Fear", "../Emotion Audio/07_disgust","../Emotion Audio/08_surprised"], 1.0,1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "emotion", False)


print(aT.fileClassification("../Emotion Audio/Testing data/Actor_22/03-01-01-01-01-01-22.wav", "emotion","svm"))


def get_files_in_directory(dir, file_extension=".wav",):

    directory = os.fsencode(dir)
    files = []

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(file_extension):
            dir_and_file =os.path.join(directory, filename)
            print(dir_and_file)
            files += dir_and_file
            continue
        else:
            continue

    return files

def classify_dir(dir,trained_machine_name,trained_machine_algorithm, file_extension=".wav"):
    location_of_files=[]
    location_of_files = get_files_in_directory(dir,file_extension)

    for file in location_of_files:
        print(aT.fileClassification(file,trained_machine_name,trained_machine_algorithm))

