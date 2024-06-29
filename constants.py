import os

GLOBAL_PATH = os.path.dirname(os.path.abspath(__file__))

MACKEY_GLASS_DATASET_PATH = "{}/data/MackeyGlass/MackeyGlass_t17.txt".format(
    GLOBAL_PATH
)

LASTFM_DATASET_PATH = (
    r"C:\Users\a\Desktop\arno\atesn\time-lstm\time_lstm-master\preprocess\data\music"
)
LASTFM_USER_ITEMS = r"{}\user-item.lst".format(LASTFM_DATASET_PATH)

CAVE_DATASET_PATH = "{}/data/Speleothem/cave-data.txt".format(GLOBAL_PATH)

REGULAR_UNEMPOYMENT_RATE_DATASET_PATH = (
    "{}/data/Regular/UnemploymentRate/dataset.txt".format(GLOBAL_PATH)
)

# UWaveALLGestureDATASETPATH = "{}\\data\\UWaveGesture".format(GLOBAL_PATH)
UWaveALLGestureDATASETPATH = os.path.join(GLOBAL_PATH, "data", "UWaveGesture")
