from pyAudioAnalysis import audioTrainTest as aT
import threading
import os

EMOTIONS = {"0.0": "Neutral", "1.0": "Calm", "2.0": "Happy", "3.0": "Sad", "4.0": "Angry", "5.0": "Fear",
            "6.0": "Disgust", "7.0": "Surprised"}


EMOTIONS_LIST = ["Neutral","Calm","Happy","Sad","Angry","Fear","Disgust","Surprised"]
from multiprocessing.pool import ThreadPool


def get_files_in_directory(dir, file_extension=".wav",):
    """
    Gets all the files in a specified directory


    :param dir: the name of the directory where the files are located
    :param file_extension: the type of files in the directory. default wav
    :return: list of file_names and their paths
    """

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
    """
    This classifies every file within a specified directory and prints / writes results.


    :param dir: Directory that we want every file (.wav) to be specified
    :param trained_machine_name: The name of the machine that has been trained
    :param trained_machine_algorithm: The type of algorithm used to train the machine (knn,svm,extratrees,gradientboosting,randomforest)
    :param file_extension: the types of files being read. default files are .wav type
    :return: void
    """

    #get all files in the directory
    files_in_directory = get_files_in_directory(dir,file_extension)

    #clear old file
    with open(trained_machine_algorithm + "-results"+ ".txt", "w") as f:
        f.write("")

    #counts the number of correctly predicted emotions
    correct = 0

    #loop through all the files in the directory
    for file in files_in_directory:

        #classify the .wav file
        #dominate_emotion: dominate emotion in classification
        dominate_emotion, emotion_statistics, emotion_paths =aT.fileClassification(file,trained_machine_name,trained_machine_algorithm)

        #make sure dominate_emotion has tenth location then convert to string (this is used when finding hte key in the EMOTIONS map)
        dominate_emotion= str(format(dominate_emotion,'.1f'))

        #Conver to list
        emotion_statistics = list(emotion_statistics)

        #convert to list
        emotion_paths = list(emotion_paths)

        #put the results into a readible format
        dominate_emotion_result, emotions_list_result = extract_results(dominate_emotion,emotion_statistics,emotion_paths)

        #extract the expected emotion from the file name
        (modality, vocal_channel, expected_emotion, emotional_intensity,statement,repitition, actor) = get_expected_emotion(file)

        #make sure expected_emotion has tenth location then convert to string (this is used when finding hte key in the EMOTIONS map)
        expected_emotion = str(format(int(expected_emotion),'.1f'))

        expected_emotion = EMOTIONS.get(expected_emotion)

        (file, file) = file.split('\\')
        with open(trained_machine_algorithm + "-results" +".txt","a+") as f:


            print("Expected: " + expected_emotion)
            print(dominate_emotion_result)

            f.write("modality-voiceChannel-emotion-emotionalIntensity-statement-repetition-actor \n\n")
            f.write("1 = neutral, 2 = calm, 3 = happy, 4 = sad, 5 = angry, 6 = fear, 7 = disgust, 8 = surprised \n")
            f.write(file + "\n")
            f.write("File results: " + trained_machine_algorithm + ".txt" + "\n")
            f.write("Expected: " + expected_emotion +"\n")
            f.write(dominate_emotion_result + "\n")

            for emotion in emotions_list_result:
                print(emotion)
                f.write(emotion + "\n")
            f.write("\n")


        if expected_emotion == EMOTIONS.get(dominate_emotion):
            print("here")
            correct += 1





    with open(trained_machine_algorithm + "-results" + ".txt", "a+") as f:
        print("Correct classifications: " + str(correct) + " Out of " + str(len(files_in_directory)))
        f.write("Correct classifications: " + str(correct) + " Out of " + str(len(files_in_directory)))

    print("File results: " + trained_machine_algorithm + ".txt")

def extract_results(dominate_emotion, emotion_statistics,emotion_paths):
    """
    Extracts the results from data fields obtained from a classification.

    The purpose of this function is to make sure that the data is in a readible format to be used in printing or writing to a file

    :param dominate_emotion: The dominate emotion (int) of the classification.
    :param emotion_statistics: The values of each emotion
    :param emotion_paths: the file path of eats file
    :return: dominate_emotion_results which is the formated dominate result, emotion_list_result is the list of emotions with their percentege of classification
    """

    #gets the dominate emotion into a readible format
    dominate_emotion_result =("Dominate Emotion: " + EMOTIONS.get(dominate_emotion))


    emotions_list_result = []
    i = 0

    for emotion in EMOTIONS_LIST:

        emotions_list_result.insert(i,str(emotion + ": " + emotion_statistics[i].astype('str')) )

        i+= 1

    return dominate_emotion_result,emotions_list_result



def get_expected_emotion(filename):
    """


    Extracts the expected emotion results from the filename.

    each filename has the following naming convention:

    FOR example:
    02-01-06-01-02-01-12.mp4

    Video-only (02)
    Speech (01)
    Fearful (06)
    Normal intensity (01)
    Statement "dogs" (02)
    1st Repetition (01)
    12th Actor (12)
    Female, as the actor ID number is even.

    -Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    -Vocal channel (01 = speech, 02 = song).
    -Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    -Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    -Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    -Repetition (01 = 1st repetition, 02 = 2nd repetition).
    -Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

    :param filename:
    :return: modality,vocal_channel,emotion,emotional_intensity,statement,repitition,actor  NOTE: see above for their definitions
    """

    # -Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    # -Vocal channel (01 = speech, 02 = song).
    # -Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    # -Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    # -Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    # -Repetition (01 = 1st repetition, 02 = 2nd repetition).
    # -Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).
    (modality, vocal_channel,emotion,emotional_intensity,statement,repitition, actor)=filename.split("-")

    #emotion numbers are off by one.  file has it 1 - 8 but in the program it is 0.0 - 7.0


    emotion = int(emotion) - 1

    #convert emotion back to string
    emotion = str(emotion)



    return modality,vocal_channel,emotion,emotional_intensity,statement,repitition,actor

def main():


    # aT.featureAndTrain(["../Emotion Audio/01_Neatral","../Emotion Audio/02_Calm","../Emotion Audio/03_Happy","../Emotion Audio/04_sad", "../Emotion Audio/05_angry","../Emotion Audio/06_Fear", "../Emotion Audio/07_disgust","../Emotion Audio/08_surprised"], 1.0,1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "emotionKNN", False)
    # aT.featureAndTrain(["../Emotion Audio/01_Neatral","../Emotion Audio/02_Calm","../Emotion Audio/03_Happy","../Emotion Audio/04_sad", "../Emotion Audio/05_angry","../Emotion Audio/06_Fear", "../Emotion Audio/07_disgust","../Emotion Audio/08_surprised"], 1.0,1.0, aT.shortTermWindow, aT.shortTermStep, "randomforest", "emotionRandomforest", False)
    # aT.featureAndTrain(["../Emotion Audio/01_Neatral","../Emotion Audio/02_Calm","../Emotion Audio/03_Happy","../Emotion Audio/04_sad", "../Emotion Audio/05_angry","../Emotion Audio/06_Fear", "../Emotion Audio/07_disgust","../Emotion Audio/08_surprised"], 1.0,1.0, aT.shortTermWindow, aT.shortTermStep, "gradienboosting", "emotionGradientBoosting", False)
    # aT.featureAndTrain(["../Emotion Audio/01_Neatral","../Emotion Audio/02_Calm","../Emotion Audio/03_Happy","../Emotion Audio/04_sad", "../Emotion Audio/05_angry","../Emotion Audio/06_Fear", "../Emotion Audio/07_disgust","../Emotion Audio/08_surprised"], 1.0,1.0, aT.shortTermWindow, aT.shortTermStep, "extratrees", "emotionExtraTrees", False)

    # print(get_files_in_directory("..\\Emotion Audio\\Testing data\\Actor_22",))

    # aT.featureAndTrain(["../Emotion Audio/01_Neatral","../Emotion Audio/02_Calm","../Emotion Audio/03_Happy","../Emotion Audio/04_sad", "../Emotion Audio/05_angry","../Emotion Audio/06_Fear", "../Emotion Audio/07_disgust","../Emotion Audio/08_surprised"], 5.0,5.0, aT.shortTermWindow, aT.shortTermStep, "svm", "emotion5point0", False)


    # print(aT.fileClassification("../Emotion Audio/Testing data/Actor_22/03-01-01-01-01-01-22.wav", "emotion","svm"))

    # print("svm")
    # classify_dir("../Emotion Audio/Testing data/Actor_22", "emotion5point0", "svm")
    # #
    # #
    # print("EmotionKNN")
    # #
    classify_dir("../Emotion Audio/Testing data/Actor_22", "emotionKNN", "knn")
    # #
    # print("RandomForest")
    # ## not working
    # classify_dir("../Emotion Audio/Testing data/Actor_22", "emotionRandomforest", "randomforest")
    #
    # print("gradientboosting")
    # ## not working
    # # classify_dir("../Emotion Audio/Testing data/Actor_22", "emotionGradientBoosting", "gradientboosting")
    #
    # print("extra trees")
    #
    # classify_dir("../Emotion Audio/Testing data/Actor_22", "emotionExtraTrees", "extratrees")



main()