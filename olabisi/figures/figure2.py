"""
Figure 2: PLSR model of determining if data can predict MSC vs. Dual
"""
from unittest import skip
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from ..imports import import_olabisi_hemi_xa
from ..plsr import plsr_model


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 10), (2, 3))

    # Add subplot labels
    subplotLabel(ax)
    
    olabisiXA, olabisiDF, meanDF = import_olabisi_hemi_xa(lod=True, perc_per_cyt=0.1, data="nomgh1")

    plsr_model(olabisiDF,ax[0],ax[1],ax[2], ncomp=10)
    for i in range(3):
        ax[i].set_title("Raw Data: W/O Average")
    plsr_model(meanDF,ax[3],ax[4],ax[5], ncomp=10)
    for i in range(3):
        ax[i+3].set_title("Raw Data: W/ Average")

    return f
