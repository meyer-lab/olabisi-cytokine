import tensorly as tl
import numpy as np
import seaborn as sns
import pandas as pd
from tensorly.decomposition import non_negative_parafac, parafac
from tensorly import partial_svd
from tensorly.tenalg import khatri_rao
from copy import deepcopy
from tqdm import tqdm


def tensordecomp(tensor, rank, td="perform_cp"):
    """ Takes Tensor, and mask and returns tensor factorized form. """
    if td == "NN":
        tfac = non_negative_parafac(np.nan_to_num(tensor), rank=rank, mask=np.isfinite(tensor), init='random', n_iter_max=5000, tol=1e-9, random_state=1)
    elif td == "perform_cp":
        tfac = perform_CP(tensor, r=rank, tol=1e-6, maxiter=5000, progress=False, callback=None)
    else:
        tfac = parafac(np.nan_to_num(tensor), rank=rank, mask=np.isfinite(tensor), init='random', n_iter_max=5000, tol=1e-9, random_state=1)

    return tfac


def R2Xplot(ax, original_tensor, rank, td):
    """Creates R2X plot for non-neg CP tensor decomposition"""
    varHold = np.zeros(rank)
    for i in range(1, rank + 1):
        print("Rank:", i)
        tFac = tensordecomp(original_tensor.to_numpy(), i, td=td)
        varHold[i - 1] = calcR2X(tFac,original_tensor)

    ax.scatter(np.arange(1, rank + 1), varHold, c='k', s=20.)
    ax.set(title="R2X", ylabel="Variance Explained", xlabel="Number of Components", 
           ylim=(0, 1), xlim=(0, rank + 0.5), xticks=np.arange(0, rank + 1))


def calcR2X(tFac, tIn=None, mIn=None):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    assert (tIn is not None) or (mIn is not None)

    vTop, vBottom = 0.0, 0.0

    if tIn is not None:
        tMask = np.isfinite(tIn)
        tIn = np.nan_to_num(tIn)
        vTop += np.linalg.norm(tl.cp_to_tensor(tFac) * tMask - tIn)**2.0
        vBottom += np.linalg.norm(tIn)**2.0
    if mIn is not None:
        mMask = np.isfinite(mIn)
        recon = tFac if isinstance(tFac, np.ndarray) else buildMat(tFac)
        mIn = np.nan_to_num(mIn)
        vTop += np.linalg.norm(recon * mMask - mIn)**2.0
        vBottom += np.linalg.norm(mIn)**2.0

    return 1.0 - vTop / vBottom


def tFac_DF(X, rank, td="perform_cp"):
    """This returns the normalized factors in dataframe form."""
    # Original_tensor == X
    cp_factors  = tensordecomp(X.to_numpy(), rank, td=td)
    cmpCol = [f"Cmp. {i}" for i in np.arange(1, cp_factors.rank + 1)]
    coords = {X.dims[0]: X.coords[X.dims[0]],
              X.dims[1]: X.coords[X.dims[1]],
              X.dims[2]: X.coords[X.dims[2]],
              X.dims[3]: X.coords[X.dims[3]]}
    
    fac_df = [pd.DataFrame(cp_factors.factors[i], columns=cmpCol, index=coords[key]) for i, key in enumerate(coords)]
        
    return fac_df
    
class IterativeSVD(object):
    def __init__(
            self,
            rank,
            convergence_threshold=1e-7,
            max_iters=500,
            random_state=None,
            min_value=None,
            max_value=None,
            verbose=False):
        self.min_value=min_value
        self.max_value=max_value
        self.rank = rank
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        self.random_state = random_state

    def clip(self, X):
        """
        Clip values to fall within any global or column-wise min/max constraints
        """
        X = np.asarray(X)
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        if X.dtype != "f" and X.dtype != "d":
            X = X.astype(float)

        assert X.ndim == 2
        missing_mask = np.isnan(X)
        assert not missing_mask.all()
        return X, missing_mask

    def fit_transform(self, X, y=None):
        """
        Fit the imputer and then transform input `X`
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask
        X_filled = X_original.copy()
        X_filled[missing_mask] = 0.0
        assert isinstance(X_filled, np.ndarray)
        X_result = self.solve(X_filled, missing_mask)
        assert isinstance(X_result, np.ndarray)
        X_result = self.clip(np.asarray(X_result))
        X_result[observed_mask] = X_original[observed_mask]
        return 


def initialize_cp(tensor: np.ndarray, rank: int):
    """Initialize factors used in `parafac`.
    Parameters
    ----------
    tensor : ndarray
    rank : int
    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    factors = [np.ones((tensor.shape[i], rank)) for i in range(tensor.ndim)]
    contain_missing = (np.sum(~np.isfinite(tensor)) > 0)

    # SVD init mode whose size is larger than rank
    for mode in range(tensor.ndim):
        if tensor.shape[mode] >= rank:
            unfold = tl.unfold(tensor, mode)
            if contain_missing:
                si = IterativeSVD(rank)
                unfold = si.fit_transform(unfold)

            factors[mode] = partial_svd(unfold, rank, flip=True)[0]

    return tl.cp_tensor.CPTensor((None, factors))


def perform_CP(tOrig, r=6, tol=1e-6, maxiter=50, progress=False, callback=None):
    """ Perform CP decomposition. """
    if callback: callback.begin()
    tFac = initialize_cp(tOrig, r)

    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]

    R2X_last = -np.inf
    tFac.R2X = calcR2X(tFac, tOrig)
    if callback: callback.first_entry(tFac)

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    tq = tqdm(range(maxiter), disable=(not progress))
    for i in tq:
        # Solve on each mode
        for m in range(len(tFac.factors)):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m])

        R2X_last = tFac.R2X
        tFac.R2X = calcR2X(tFac, tOrig)
        tq.set_postfix(R2X=tFac.R2X, delta=tFac.R2X - R2X_last, refresh=False)
        assert tFac.R2X > 0.0
        if callback: callback.update(tFac)

        if tFac.R2X - R2X_last < tol:
            break

    tFac = cp_normalize(tFac)

    return tFac

def cp_normalize(tFac):
    """ Normalize the factors using the inf norm. """
    for i, factor in enumerate(tFac.factors):
        scales = np.linalg.norm(factor, ord=np.inf, axis=0)
        tFac.weights *= scales
        if i == 0 and hasattr(tFac, 'mFactor'):
            mScales = np.linalg.norm(tFac.mFactor, ord=np.inf, axis=0)
            tFac.mWeights = scales * mScales
            tFac.mFactor /= mScales

        tFac.factors[i] /= scales

    return tFac

def censored_lstsq(A: np.ndarray, B: np.ndarray, uniqueInfo=None) -> np.ndarray:
    """Solves least squares problem subject to missing data.
    Note: uses a for loop over the missing patterns of B, leading to a
    slower but more numerically stable algorithm
    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """
    X = np.empty((A.shape[1], B.shape[1]))
    # Missingness patterns
    if uniqueInfo is None:
        unique, uIDX = np.unique(np.isfinite(B), axis=1, return_inverse=True)
    else:
        unique, uIDX = uniqueInfo

    for i in range(unique.shape[1]):
        uI = uIDX == i
        uu = np.squeeze(unique[:, i])

        Bx = B[uu, :]
        X[:, uI] = np.linalg.lstsq(A[uu, :], Bx[:, uI], rcond=-1)[0]
    return X.T

def buildMat(tFac):
    """ Build the matrix in CMTF from the factors. """
    if hasattr(tFac, 'mWeights'):
        return tFac.factors[0] @ (tFac.mFactor * tFac.mWeights).T
    return tFac.factors[0] @ tFac.mFactor.T


class IterativeSVD(object):
    def __init__(
            self,
            rank,
            convergence_threshold=1e-7,
            max_iters=500,
            random_state=None,
            min_value=None,
            max_value=None,
            verbose=False):
        self.min_value=min_value
        self.max_value=max_value
        self.rank = rank
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        self.random_state = random_state

    def clip(self, X):
        """
        Clip values to fall within any global or column-wise min/max constraints
        """
        X = np.asarray(X)
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        if X.dtype != "f" and X.dtype != "d":
            X = X.astype(float)

        assert X.ndim == 2
        missing_mask = np.isnan(X)
        assert not missing_mask.all()
        return X, missing_mask

    def fit_transform(self, X, y=None):
        """
        Fit the imputer and then transform input `X`
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask
        X_filled = X_original.copy()
        X_filled[missing_mask] = 0.0
        assert isinstance(X_filled, np.ndarray)
        X_result = self.solve(X_filled, missing_mask)
        assert isinstance(X_result, np.ndarray)
        X_result = self.clip(np.asarray(X_result))
        X_result[observed_mask] = X_original[observed_mask]
        return X_result

    def _converged(self, X_old, X_new, missing_mask):
        F32PREC = np.finfo(np.float32).eps
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm_squared = (old_missing_values ** 2).sum()
        # edge cases
        if old_norm_squared == 0 or \
                (old_norm_squared < F32PREC and ssd > F32PREC):
            return False
        else:
            return (ssd / old_norm_squared) < self.convergence_threshold

    def solve(self, X, missing_mask):
        observed_mask = ~missing_mask
        X_filled = X
        for i in range(self.max_iters):
            curr_rank = self.rank
            self.U, S, V = partial_svd(X_filled, curr_rank, random_state=self.random_state)
            X_reconstructed = self.U @ np.diag(S) @ V
            X_reconstructed = self.clip(X_reconstructed)

            # Masked mae
            mae = np.mean(np.abs(X[observed_mask] - X_reconstructed[observed_mask]))

            if self.verbose:
                print(
                    "[IterativeSVD] Iter %d: observed MAE=%0.6f" % (
                        i + 1, mae))
            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstructed,
                missing_mask=missing_mask)
            X_filled[missing_mask] = X_reconstructed[missing_mask]
            if converged:
                break
        return X_filled