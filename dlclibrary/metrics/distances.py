"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import numpy as np
import pandas as pd


def pairwisedistances(DataCombined, scorer1, scorer2, pcutoff=-1, bodyparts=None):
    """Calculates the pairwise Euclidean distance metric over body parts vs. images"""
    mask = DataCombined[scorer2].xs("likelihood", level=1, axis=1) >= pcutoff
    if bodyparts == None:
        Pointwisesquareddistance = (DataCombined[scorer1] - DataCombined[scorer2]) ** 2
        RMSE = np.sqrt(
            Pointwisesquareddistance.xs("x", level=1, axis=1)
            + Pointwisesquareddistance.xs("y", level=1, axis=1)
        )  # Euclidean distance (proportional to RMSE)
        return RMSE, RMSE[mask]
    else:
        Pointwisesquareddistance = (
            DataCombined[scorer1][bodyparts] - DataCombined[scorer2][bodyparts]
        ) ** 2
        RMSE = np.sqrt(
            Pointwisesquareddistance.xs("x", level=1, axis=1)
            + Pointwisesquareddistance.xs("y", level=1, axis=1)
        )  # Euclidean distance (proportional to RMSE)
        return RMSE, RMSE[mask]



def evaluate_assembly(
