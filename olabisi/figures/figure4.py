"""
Figure 4: Investigating cytokines between treatments
"""
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from ..imports import import_olabisi_hemi_xa


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((13, 16), (8, 3))

    # Add subplot labels
    subplotLabel(ax)
    
    olabisiXA, olabisiDF, meanDF = import_olabisi_hemi_xa(lod=True, perc_per_cyt=0.1, data="nomgh1")
    
    cytokines = np.unique(meanDF.columns[3::])
    axis = 0
    
    for i, loc in enumerate(np.unique(meanDF["Location"])):
        if i == 2:
            pass
        else:
            continue
        newDF = meanDF.loc[meanDF["Location"] == loc]   
        for k, cytok in enumerate(cytokines):
            if k > 23:
                pass
            # if k <= 23:
            #     pass
            else:
                continue
            sns.lineplot(data=newDF,x="Day", y=cytok, hue="Treatment", ax=ax[axis]).set(title="Location:"+loc)
            axis += 1 
            print(k)
           
    return f
