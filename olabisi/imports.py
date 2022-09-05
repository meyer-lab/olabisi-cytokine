from optparse import Values
import xarray as xa
import numpy as np
import pandas as pd
from scipy.stats import zscore as zscoreFunc

def import_olabisi_hemi_xa(lod=False,zscore=False, perc_per_cyt=0.5):
    """"Import of the Olabisi cytokine data of aggregated dataset"""
    hemi_totalDF = pd.read_csv("olabisi/data/olabisi_hemi_edited.csv", na_values="-").reset_index(drop=True)
    # ["hMIG/CXCL9", "hMIP-1a","hMIP-1b"] had no signals for all conditions
    hemi_totalDF = hemi_totalDF.drop(["Plate", "Location","Well ID", "Sample ID", "Standard","hMIG/CXCL9", "hMIP-1a","hMIP-1b"],axis=1)

    # Renaming Dataframes
    hemi_totalDF["Group"] = hemi_totalDF["Group"].str.split("_").str[0]
    hemi_totalDF = hemi_totalDF.rename({"Group":"Location","Cell": "Treatment","Time (days)":"Day"},axis=1)
    cytokines = hemi_totalDF.columns.values
    cytokines = cytokines[3::]
    # Ensuring all values for cytokines are values
    hemi_totalDF[cytokines] = hemi_totalDF[cytokines].astype('float64')

   # Removing all rows with remove_dict
    hemi_totalDF = hemi_totalDF[hemi_totalDF["Location"] != "ctrl"]

    # Replacing NaN values with limit of detection for each cytokine
    if lod is True: 
        hemi_lodDF = pd.read_csv("olabisi/data/olabisi_hemi_lod.csv").set_index("Analyte").transpose()
        hemi_lodDF = hemi_lodDF.drop(["hMIG/CXCL9", "hMIP-1a","hMIP-1b"],axis=1).reset_index(drop=True)
        for i, cyt in enumerate(cytokines):
            hemi_totalDF[cyt] = hemi_totalDF[cyt].fillna(float(hemi_lodDF[cyt].values))
            assert np.isfinite(hemi_totalDF[cyt].values.all())
        
        if zscore is True:
            # Zscoring cytokines
            hemi_totalDF[cytokines] = hemi_totalDF[cytokines].apply(zscoreFunc, axis=1)
    else:
        if zscore is True:
            # Zscoring cytokines
            for i, cyt in enumerate(cytokines):
                hemi_totalDF[cyt] = (hemi_totalDF[cyt] - hemi_totalDF[cyt].mean())/hemi_totalDF[cyt].std()
    
    reduced_cytokines = []
    # Only keeps cytokines with specific percentage across all experiments
    for i, cyt in enumerate(cytokines):
       if 1-(np.count_nonzero(np.isnan(hemi_totalDF[cyt].values))/len(hemi_totalDF[cyt].values)) >= perc_per_cyt:
           reduced_cytokines = np.append(reduced_cytokines,cyt)      
                            
    locations = hemi_totalDF["Location"].unique()
    treatments = hemi_totalDF["Treatment"].unique()
    days = np.sort(hemi_totalDF["Day"].unique())
    
    assert np.isfinite(hemi_totalDF[cytokines].values.all())

    # Building tensor/dataframe of mean values for each experimental condition combination
    mean_olabisi_DF = pd.DataFrame([])
    tensor = np.empty((len(locations), len(treatments), len(days), len(reduced_cytokines)))
    tensor[:] = np.NaN
    
    print("Amount of Original Cytokines:",np.shape(cytokines))
    print("Amount of Reduced Cytokines:", np.shape(reduced_cytokines))
    for i, loc in enumerate(locations):
        for j, treat in enumerate(treatments):
            for k, time in enumerate(days):
                for l, mark in enumerate(reduced_cytokines):
                    conditionDF = hemi_totalDF.loc[(hemi_totalDF["Location"] == loc) & (hemi_totalDF["Treatment"] == treat) & (hemi_totalDF["Day"] == time)]
                    cytok = conditionDF[mark].values

                    # Some conditions do not have any values 
                    if cytok.shape[0] == 0:
                        mean = np.NaN
                    else:
                        # Only uses means whose values are in at 50% given of the experimental repeats
                        if 1-(np.count_nonzero(np.isnan(cytok))/len(cytok)) >= .5:
                            mean = np.nanmean(cytok)
                        else:
                            mean = np.NaN
                            
                    # Obtaining average for every condition in to a DataFrame
                    mean_olabisi_DF = pd.concat([mean_olabisi_DF, pd.DataFrame({"Location": loc, "Treatment": treat, "Day": time, "Cytokine": mark,
                                                                                "Mean":[mean]})])

                    tensor[i,j,k,l] = mean
    
    mean_olabisi_DF = mean_olabisi_DF.reset_index(drop=True) 
    # Shape of Tensor: Location, Treatment, Day, and Cytokine       
    olabisiXA = xa.DataArray(tensor, dims=("Location", "Treatment", "Day", "Cytokine"), coords={"Location": locations, "Treatment": treatments,
                                            "Day": days, "Cytokine": reduced_cytokines})

    return olabisiXA , mean_olabisi_DF, hemi_totalDF