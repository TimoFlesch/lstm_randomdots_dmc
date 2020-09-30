import numpy as np

# basic parameters
COL_BLACK = (0, 0, 0)
COL_WHITE = (255, 255, 255)


WINDOW_WIDTH = 48
WINDOW_HEIGHT = 48
WINDOW_NAME = "DMC Task"
WINDOW_COLOUR = COL_BLACK



# time
TICK_RATE = 30
TIME_FIX = 0  # frames
TIME_ISI = 15
TIME_RDK = 15
TIME_ITI = 60

SEQ_LENGTH = TIME_FIX+TIME_ISI+TIME_RDK*2
# dot parameters
N_DOTS = 50 #100         # max num of simultaneously displayed dots
DOT_SIZE = 2 #2         # size in pixels
DOT_SPEED = 5        # speed in pixels per frame

DOT_ANGLES = np.asarray([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5])
DOT_CATEGORIES = np.asarray([[0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 1, 0]])
DOT_BOUNDDISTS = np.asarray([[1, 2, 2, 1, 1, 2, 2, 1], [1, 1, 2, 2, 1, 1, 2, 2], [2, 1, 1, 2, 2, 1, 1, 2], [2, 2, 1, 1, 2, 2, 1, 1]])
DOT_BOUNDANGLES = np.asarray([180, 45, 90, 135])
DOT_BOUNDIDX = 1
DOT_CATLABELS = DOT_CATEGORIES[DOT_BOUNDIDX-1, :]
DOT_BOUNDARY = DOT_BOUNDANGLES[DOT_BOUNDIDX-1]

DOT_REPETITIONS = 10  # how many repetitions of same trials?
DOT_COHERENCE = 1  # motion coherence (between 0 and 1)
DOT_COLOR = COL_WHITE



# aperture parameters
APERTURE_RADIUS = 40       # radius in pixels
APERTURE_WIDTH = 2         # line width in pixels
APERTURE_COLOR = COL_WHITE

# fixation parameters
FIX_SIZE = (5, 5)  # width and height of fix cross
FIX_COLOR = COL_WHITE
FIX_WIDTH = 2  # line width
