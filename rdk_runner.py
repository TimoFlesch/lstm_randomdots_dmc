# random dot motion kinematogram
# timo flesch, 2019


# import matplotlib.pyplot as plt
# from array2gif import write_gif

import rdktools.rdk_params as params
from rdktools.rdk_experiment import set_trials, DMCTrial
import tensorflow as tf
import pygame

def main():
    trials = set_trials(n_reps=params.DOT_REPETITIONS,
                        angles=params.DOT_ANGLES)
    # instantiate task
    dmc = DMCTrial()

    # instantiate neural netowrk
    # nnet = Agent()
    trials = dmc.set_trialids()
    ii_train = 0
    for ii in range(trials.shape[0]):
            # sample orientations and category label
            # TODO: have vect of all combinations and label (n*3)
            # angle_sample, angle_probe, cat_label = dmc.get_io()
            # run trial
            frames = dmc.run(trials[ii,0],trials[ii,1])

            # feed frames into network and update weights
            # y_out = nnet.train(frames)

            # store output


            # if ii_train % testsess, run evaluation on test set, store activity patterns

    pygame.quit()
    quit()


if __name__ == "__main__":
    main()
