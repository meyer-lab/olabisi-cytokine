"""
Figure 3: Investigating cytokines signals of raw/mean data
"""
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from ..imports import import_olabisi_hemi_xa


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((13, 16), (4, 3))

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
        print("Location:", loc)
        for j, treat in enumerate(np.unique(meanDF["Treatment"])):
            newDF = meanDF.loc[(meanDF["Location"] == loc) & (meanDF["Treatment"] == treat)]   
            for k in range(int(len(cytokines)/6)):
                diffDF = newDF[np.append("Day",cytokines[6*k:6*(k+1)])].melt("Day", var_name="Cytokines", value_name="Vals")
                sns.lineplot(data=diffDF,x="Day", y="Vals", hue="Cytokines", ax=ax[axis]).set(title="Treatment:"+treat+"-"+"Location:"+loc)
                axis += 1
                
    return f
