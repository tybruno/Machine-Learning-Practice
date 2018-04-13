from pyAudioAnalysis import audioTrainTest as aT
import threading
import os

EMOTIONS = {"0.0": "Neutral", "1.0": "Calm", "2.0": "Happy", "3.0": "Sad", "4.0": "Angry", "6.0": "Fear",
            "7.0": "Disgust", "8.0": "Surprised"}
EMOTIONS_LIST = ["Neutral","Calm","Happy","Sad","Angry","Fear","Disgust","Surprised"]
from multiprocessing.pool import ThreadPool


def get_files_in_directory(dir, file_extension=".wav",):

    directory = os.fsencode(dir)
    directory = directory.decode("utf-8")
    files = []
    i = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if filename.endswith(file_extension):
            dir_and_file =  os.path.join(directory, filename)
            # print(dir_and_file)
            files.insert(i ,dir_and_file)
            i+= 1
            continue
        else:
            continue

    return files

def classify_dir(dir,trained_machine_name,trained_machine_algorithm, file_extension=".wav"):
    location_of_files=[]
    location_of_files = get_files_in_directory(dir,file_extension)



    for file in location_of_files:
        print(file)
        aT.fileClassification(file,trained_machine_name,trained_machine_algorithm)

def print_classification_results(dominate_emotion, emotion_statistics, emotion_paths):
    print("Dominate Emotion: "+ EMOTIONS.get(dominate_emotion.astype('str')))


    print(emotion_statistics)
    print(EMOTIONS_LIST)

    i = 0
    array = []
    map = {}

    for emotion in EMOTIONS_LIST:
        print(emotion + ": " + emotion_statistics[i].astype('str'))
        # array[i] = emotion + ": " + emotion_statistics[i].astype('str')
        i+=1


    # for i, j in EMOTIONS_LIST,emotion_statistics:
    #     print(i + j)
    # for value in emotion_statistics:
    #
    #     print(EMOTIONS_LIST[i] + ": " + value.astype('str'))
    #
    #     i+= 1
    #     # print(i)
    #     # print(value)
    #     # print(EMOTIONS.get(str(i)) + ": " + str(value.astype('str')))
    #     # i+= 1.0

def main():
    EMOTIONS = {"0.0":"Neutral", "1.0":"Calm","2.0":"Happy","3.0":"Sad","4.0":"Angry","6.0":"Fear", "7.0":"Disgust","8.0":"Surprised"}

    # classify_dir("..\\Emotion Audio\\Testing data\\Actor_22","emotion","svm")
    # print(get_files_in_directory("..\\Emotion Audio\\Testing data\\Actor_22",))

    # aT.featureAndTrain(["../Emotion Audio/01_Neatral","../Emotion Audio/02_Calm","../Emotion Audio/03_Happy","../Emotion Audio/04_sad", "../Emotion Audio/05_angry","../Emotion Audio/06_Fear", "../Emotion Audio/07_disgust","../Emotion Audio/08_surprised"], 1.0,1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "emotion", False)


    # print(aT.fileClassification("../Emotion Audio/Testing data/Actor_22/03-01-01-01-01-01-22.wav", "emotion","svm"))
    dominate_emotion, emotion_statistics, emotion_paths=aT.fileClassification("../Emotion Audio/Testing data/Actor_22/03-01-01-01-01-01-22.wav", "emotion", "svm")
    print(emotion_statistics)
    print(emotion_paths)
    print_classification_results(dominate_emotion,emotion_statistics,emotion_paths)



main()