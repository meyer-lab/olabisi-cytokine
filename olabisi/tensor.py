import tensorly as tl
import numpy as np
import seaborn as sns
import pandas as pd
from tensorly.decomposition import non_negative_parafac, parafac


def tensordecomp(tensor, rank, nn=False):
    """ Takes Tensor, and mask and returns tensor factorized form. """
    if nn:
        tfac = non_negative_parafac(np.nan_to_num(tensor), rank=rank, mask=np.isfinite(tensor), init='random', n_iter_max=5000, tol=1e-9, random_state=1)
    else:
        tfac = parafac(np.nan_to_num(tensor), rank=rank, mask=np.isfinite(tensor), init='random', n_iter_max=5000, tol=1e-9, random_state=1)

    return tfac


def R2Xplot(ax, original_tensor, rank):
    """Creates R2X plot for non-neg CP tensor decomposition"""
    varHold = np.zeros(rank)
    for i in range(1, rank + 1):
        print(i)
        tFac = tensordecomp(original_tensor.to_numpy(), i)
        varHold[i - 1] = calcR2X(original_tensor, tFac)

    ax.scatter(np.arange(1, rank + 1), varHold, c='k', s=20.)
    ax.set(title="R2X", ylabel="Variance Explained", xlabel="Number of Components", 
           ylim=(0, 1), xlim=(0, rank + 0.5), xticks=np.arange(0, rank + 1))


def calcR2X(original_tensor, tensorFac):
    """ Calculate R2X """
    tErr = np.nanvar(tl.cp_to_tensor(tensorFac) - original_tensor)
    return 1.0 - tErr / np.nanvar(original_tensor)


def tFac_DF(X, rank, nn=False):
    """This returns the normalized factors in dataframe form."""
    # Original_tensor == X
    tFac = tensordecomp(X.to_numpy(), rank, nn=False)
    cp_factors = tl.cp_normalize(tFac)
    cmpCol = [f"Cmp. {i}" for i in np.arange(1, cp_factors.rank + 1)]
    coords = {X.dims[0]: X.coords[X.dims[0]],
              X.dims[1]: X.coords[X.dims[1]],
              X.dims[2]: X.coords[X.dims[2]],
              X.dims[3]: X.coords[X.dims[3]]}
    
    fac_df = [pd.DataFrame(cp_factors.factors[i], columns=cmpCol, index=coords[key]) for i, key in enumerate(coords)]
        
    return fac_df
    

