import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr
   
def plsr_model(olabisiDF,ax1, ax2, ax3, ncomp):
    """PLSR to determine if able to predict MSC vs Dual"""
    # dayDF = olabisiDF.loc[olabisiDF["Day"] == 10]
    dayDF = olabisiDF
    dayDF = dayDF.replace({"MSC": 0, "Dual": 1}) # Converting treatments to values
    dayDF = dayDF.drop(["Location","Day"],axis=1)
  
    X = dayDF.iloc[:,1::].to_numpy() # Getting values for all cytokines
    Y = dayDF.iloc[:,0]
    
    # Determing variance explained R2X
    arr = np.arange(1,ncomp+1,1)
    scoresmatrix = np.zeros([len(arr)])
    for i in range(len(arr)):
        pls = PLSRegression(n_components=arr[i],scale=True)
        pls.fit_transform(X,Y)
        scoresmatrix[i] = pls.score(X,Y)
    
    ax1.scatter(arr,scoresmatrix)
    ax1.set_xlabel("Principal Components")
    ax1.set_ylabel("R2X")

    # Determining loadings for X and Y
    pls2 = PLSRegression(n_components=ncomp,scale=True)
    pls2.fit_transform(X,Y)
    xloadings2 = pls2.x_loadings_
    yloadings2 = pls2.y_loadings_
 
    ax2.scatter(xloadings2[:,0],xloadings2[:,1],label="X Loadings")
    ax2.scatter(yloadings2[:,0],yloadings2[:,1],label="Y Loadings")

    ax2.legend(loc = 'upper left')
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    
    # Calculating R^2 Value from Pearson Correlation 
    prediction = pls2.predict(X)
    [r,pvalue] = pearsonr(Y,prediction)
    print("R^2 Value:",np.square(r))
    ax3.scatter(Y,prediction)
    ax3.set_xlabel("Observed")
    ax3.set_ylabel("Predicted")
    
    return 
    