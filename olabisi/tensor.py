import numpy as np
import pandas as pd
from tensorpack import perform_CP


def tensordecomp(tensor, rank):
    """Takes Tensor, and mask and returns tensor factorized form."""
    return perform_CP(tensor, r=rank, tol=1e-7, maxiter=5000)


def R2Xplot(ax, original_tensor, rank):
    """Creates R2X plot for non-neg CP tensor decomposition"""
    varHold = np.zeros(rank)
    for i in range(1, rank + 1):
        print("Rank:", i)
        tFac = tensordecomp(original_tensor.to_numpy(), i)
        varHold[i - 1] = tFac.R2X

    ax.scatter(np.arange(1, rank + 1), varHold, c="k", s=20.0)
    ax.set(
        title="R2X",
        ylabel="Variance Explained",
        xlabel="Number of Components",
        ylim=(0, 1),
        xlim=(0, rank + 0.5),
        xticks=np.arange(0, rank + 1),
    )


def tFac_DF(X, rank):
    """This returns the normalized factors in dataframe form."""
    # Original_tensor == X
    cp_factors = tensordecomp(X.to_numpy(), rank)
    cmpCol = [f"Cmp. {i}" for i in np.arange(1, cp_factors.rank + 1)]
    coords = {
        X.dims[0]: X.coords[X.dims[0]],
        X.dims[1]: X.coords[X.dims[1]],
        X.dims[2]: X.coords[X.dims[2]],
        X.dims[3]: X.coords[X.dims[3]],
    }

    fac_df = [
        pd.DataFrame(cp_factors.factors[i], columns=cmpCol, index=coords[key])
        for i, key in enumerate(coords)
    ]

    return fac_df
