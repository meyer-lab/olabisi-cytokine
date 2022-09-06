import numpy as np
import pandas as pd


def import_olabisi_hemi_xa(lod=False, zscore=False, perc_per_cyt=0.1):
    """ "Import of the Olabisi cytokine data of aggregated dataset"""
    hemi_totalDF = pd.read_csv(
        "olabisi/data/olabisi_hemi_edited.csv", na_values="-"
    ).reset_index(drop=True)
    hemi_totalDF = hemi_totalDF.drop(
        ["Plate", "Location", "Well ID", "Sample ID", "Standard"], axis=1
    )

    # Renaming Dataframes
    hemi_totalDF["Group"] = hemi_totalDF["Group"].str.split("_").str[0]
    hemi_totalDF = hemi_totalDF.rename({"Group": "Location"}, axis=1)
    cytokines = hemi_totalDF.columns.values
    cytokines = cytokines[3::]
    # Ensuring all values for cytokines are values
    hemi_totalDF[cytokines] = hemi_totalDF[cytokines].astype("float64")

    # Removing all rows with string
    hemi_totalDF = hemi_totalDF[hemi_totalDF["Location"] != "ctrl"]

    # Replacing NaN values with limit of detection for each cytokine
    if lod is True:
        hemi_lodDF = (
            pd.read_csv("olabisi/data/olabisi_hemi_lod.csv")
            .set_index("Analyte")
            .transpose()
        )
        hemi_lodDF = hemi_lodDF.reset_index(drop=True)
        for cyt in cytokines:
            hemi_totalDF[cyt] = hemi_totalDF[cyt].fillna(float(hemi_lodDF[cyt].values))
            assert np.isfinite(hemi_totalDF[cyt].values.all())

    if zscore is True:
        # Zscoring cytokines
        for cyt in cytokines:
            hemi_totalDF[cyt] = (
                hemi_totalDF[cyt] - hemi_totalDF[cyt].mean()
            ) / hemi_totalDF[cyt].std()

    print("Amount of Original Cytokines:", np.shape(cytokines))

    # Reshape to tensor
    gcol = ["Location", "Treatment", "Day"]
    hemi_meanDF = hemi_totalDF.groupby(gcol).mean()
    olabisiXA = hemi_meanDF.to_xarray()
    olabisiXA = olabisiXA.to_array(dim="Cytokines")

    print(olabisiXA.shape)

    # Remove mostly missing cytokines
    olabisiXA = olabisiXA.loc[
        np.mean(np.isfinite(olabisiXA), axis=(1, 2, 3)) >= perc_per_cyt, :, :, :
    ]

    print(olabisiXA.shape)

    return olabisiXA, hemi_totalDF
