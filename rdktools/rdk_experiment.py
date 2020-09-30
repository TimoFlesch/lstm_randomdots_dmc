# random dot motion kinematogram
# timo flesch, 2019

import pygame
from pygame.locals import *
import random
import numpy as np
import rdktools.rdk_params as params
import os, sys

# set SDL to use the dummy NULL video driver,
#   so it doesn't need a windowing system.
os.environ["SDL_VIDEODRIVER"] = "dummy"
from rdktools.rdk_stimuli import RDK, Fixation, BlankScreen


def set_trials(n_reps=10, angles=[0, 90, 135], shuff=True):
    """ creates vector of all motion directions """
    all_trials = np.array([])

    for thisAngle in angles:
        all_trials = np.append(all_trials, np.repeat(thisAngle, n_reps),
                                axis=0)

    if shuff:
        random.shuffle(all_trials)

    return all_trials


class DMCTrial(object):
    """defines the sequence of events within a trial.
       returns a n-D matrix with all displayed frames as greyscale images (2D)
    """

    def __init__(self,angles=params.DOT_ANGLES,
                    catbound=params.DOT_BOUNDARY,
                    catlabels=params.DOT_CATLABELS):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode((params.WINDOW_WIDTH,
                            params.WINDOW_HEIGHT))
        pygame.display.set_caption(params.WINDOW_NAME)
        self.display.fill(params.COL_BLACK)
        self.angles = angles
        self.catbound = catbound
        self.categories = catlabels
        self.combinations = self.set_trialids()
        self.rdk = RDK(self.display)
        # self.fix = Fixation(self.display, f_duration=params.TIME_FIX)
        self.isi = Fixation(self.display, f_duration=params.TIME_ISI)
        # self.iti = BlankScreen(self.display, duration=params.TIME_ITI)

    def run(self, angle_sample=90, angle_probe=180):
        # frames_fix = self.fix.show()
        self.rdk.new_sample(angle_sample)
        frames_sample = self.rdk.show()
        frames_isi = self.isi.show()
        self.rdk.new_sample(angle_probe)
        frames_probe = self.rdk.show()
        # frames_iti = self.iti.show()

        return np.concatenate((frames_sample, frames_isi,
                                frames_probe), axis=0)

    def set_trialids(self):
        trial_ids = np.array([])
        trial_ids = [[[self.angles[ii],self.angles[jj],self.categories[ii],
                self.categories[jj],self.categories[ii]==self.categories[jj]]
                for ii in range(self.angles.size)] for jj in range(self.angles.size)]
        trial_ids = np.asarray(trial_ids)
        trial_ids = np.reshape(trial_ids, (self.angles.size**2,5))
        return trial_ids
