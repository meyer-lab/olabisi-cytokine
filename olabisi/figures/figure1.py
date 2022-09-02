"""
Figure 1
"""
import seaborn as sns
from .common import subplotLabel, getSetup
from ..imports import import_olabisi_hemi_xa
from ..tensor import R2Xplot, tFac_DF, tensordecomp

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    # ax, f = getSetup((10, 13), (2, 3),multz={9: 2})
    ax, f = getSetup((12, 14), (2, 3))

    # Add subplot labels
    subplotLabel(ax)
    ax[5].axis("off")
    
    olabisiXA, _, _ = import_olabisi_hemi_xa()
    R2Xplot(ax[0], olabisiXA, rank=3)
    fac_df = tFac_DF(X=olabisiXA, rank=3, nn=False)
    
    for i in range(0, 4):
        """Plots tensor factorization of cells"""
        sns.heatmap(data=fac_df[i], ax=ax[i + 1])

    return f