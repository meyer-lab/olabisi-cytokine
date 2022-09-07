"""
Figure 1
"""
import numpy as np
import seaborn as sns
from .common import subplotLabel, getSetup
from ..imports import import_olabisi_hemi_xa
from ..tensor import R2Xplot, tFac_DF


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 16), (2, 3))

    # Add subplot labels
    subplotLabel(ax)
    ax[5].axis("off")
    
    olabisiXA, _ = import_olabisi_hemi_xa()
 
    for i, days in enumerate(olabisiXA["Day"].values):
        print("Day:", days, "Missing Values Percentange:", np.mean(np.isnan(olabisiXA.isel(Day=i).values)))
    print("Total Missing Values Percentange:", np.mean(np.isnan(olabisiXA)))  
    
    R2Xplot(ax[0], olabisiXA, rank=5)
    fac_df = tFac_DF(X=olabisiXA, rank=4)

    for i in range(0, 4):
        """Plots tensor factorization of cells"""
        sns.heatmap(data=fac_df[i], ax=ax[i + 1],
            cmap=sns.diverging_palette(240, 10, as_cmap=True),)

    return f
