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
    ax, f = getSetup((12, 14), (2, 3))

    # Add subplot labels
    subplotLabel(ax)
    ax[5].axis("off")
    
    olabisiXA, olabisiDF, totalDF = import_olabisi_hemi_xa(lod=False,zscore=True,min_perc_exp_vals=0.1)
    rank = 10
    print("Missing Values Percentange:",olabisiDF["Mean"].isnull().sum()/len(olabisiDF["Mean"]))
    
    R2Xplot(ax[0], olabisiXA, rank=rank, td="perform_cp")
    fac_df = tFac_DF(X=olabisiXA, rank=rank, td="perform_cp")
    
    for i in range(0, 4):
        """Plots tensor factorization of cells"""
        sns.heatmap(data=fac_df[i], ax=ax[i + 1], cmap=sns.diverging_palette(240, 10,as_cmap=True))


    return f