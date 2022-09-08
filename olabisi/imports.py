import numpy as np
import pandas as pd


def import_olabisi_hemi_xa(lod=True, perc_per_cyt=0.1):
    """ "Import of the Olabisi cytokine data of aggregated dataset"""
    hemi_totalDF = pd.read_csv(
        "olabisi/data/olabisi_hemi_edited.csv", na_values="-"
    ).reset_index(drop=True)
    hemi_totalDF = hemi_totalDF.drop(
        ["Plate", "Location", "Well ID"], axis=1)
    # Replacing days of the experiments
    rename_days = [5, 12, 14, 16, 21, 23, 26, 28, 31]
    new_days = [6, 13, 13, 15, 20, 22, 27, 29, 30]
    hemi_totalDF = hemi_totalDF[hemi_totalDF["Day"] != 24]
    hemi_totalDF = hemi_totalDF[hemi_totalDF["Day"] != 33]
    # Renaming Dataframes
    hemi_totalDF = hemi_totalDF.replace(rename_days, new_days)
    hemi_totalDF["Group"] = hemi_totalDF["Group"].str.split("_").str[0]
    hemi_totalDF = hemi_totalDF.rename({"Group": "Location"}, axis=1)
    
    # Ensuring all values for cytokines are values
    cytokines = hemi_totalDF.columns.values
    cytokines = cytokines[3::]
    hemi_totalDF[cytokines] = hemi_totalDF[cytokines].astype("float64")

    # Removing all rows with string
    hemi_totalDF = hemi_totalDF[hemi_totalDF["Location"] != "ctrl"]
    
    for i in hemi_totalDF["Location"].unique():
        for j in hemi_totalDF["Treatment"].unique():
            timeDF = hemi_totalDF.loc[(hemi_totalDF["Location"] == i) & (hemi_totalDF["Treatment"] == j)]
            print("Location:", i, "Treatment:", j , "Time:", timeDF["Day"].unique())
                
    # Replacing NaN values with limit of detection for each cytokine
    if lod is True:
        hemi_lodDF = (pd.read_csv("olabisi/data/olabisi_hemi_lod.csv")
                        .set_index("Analyte").transpose())
        hemi_lodDF = hemi_lodDF.reset_index(drop=True)
        for cyt in cytokines:
            hemi_totalDF[cyt] = hemi_totalDF[cyt].fillna(float(hemi_lodDF[cyt].values))
            assert np.isfinite(hemi_totalDF[cyt].values.all())

    # Log transform
    hemi_totalDF[cytokines] = np.log(hemi_totalDF[cytokines])
    
    # Substracting across rows to account for dilution  
    row_mean = np.reshape(hemi_totalDF[cytokines].mean(axis=1).values, (-1, 1))
    hemi_totalDF[cytokines] = hemi_totalDF[cytokines].sub(row_mean)
    
    # Substracting by arithmetic mean
    for cyt in cytokines:
        hemi_totalDF[cyt] = (hemi_totalDF[cyt] - hemi_totalDF[cyt].mean()) / hemi_totalDF[cyt].std()
    
    # Reshape to tensor
    gcol = ["Location", "Treatment", "Day"]
    hemi_meanDF = hemi_totalDF.groupby(gcol).mean()
    olabisiXA = hemi_meanDF.to_xarray()
    olabisiXA = olabisiXA.to_array(dim="Cytokines")

    # Remove mostly missing cytokines
    olabisiXA = olabisiXA.loc[np.mean(np.isfinite(olabisiXA), axis=(1, 2, 3)) >= 
                                perc_per_cyt, :, :, :]

    print(olabisiXA.shape)

    return olabisiXA, hemi_totalDF
