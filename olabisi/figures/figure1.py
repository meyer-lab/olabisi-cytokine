"""
Figure 1
"""
from .common import subplotLabel, getSetup
from ..imports import import_olabisi_hemi_xa

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((8, 4), (1, 3))

    # Add subplot labels
    subplotLabel(ax)
    
    print(import_olabisi_hemi_xa())




    return f