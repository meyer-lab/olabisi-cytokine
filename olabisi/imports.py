
import xarray as xa
import numpy as np
import seaborn as sns
import pandas as pd

def import_olabisi_hemi_xa():
    """"Import of the Olabisi cytokine data of aggregated dataset"""
    # str = unicode(str, errors='replace')
    hemi_totalDF = pd.read_csv("olabisi/olabisi_hemi_edited.csv").reset_index(drop=True)
    # ["hMIG/CXCL9", "hMIP-1a","hMIP-1b"] had no signals for all conditions
    hemi_totalDF = hemi_totalDF.drop(["Plate", "Location","Well ID", "Sample ID", "Standard","hMIG/CXCL9", "hMIP-1a","hMIP-1b"],axis=1)

    rename_dict = {"mgh1_dual_03": "MGH1",
               "mgh1_dual_06": "MGH1",
               "mgh1_dual_08": "MGH1",
               "mgh1_dual_10": "MGH1",
               "mgh1_dual_03": "MGH1",
               "mgh1_dual_13": "MGH1",
               "mgh1_dual_15": "MGH1",
               "mgh1_dual_17": "MGH1",
               "mgh1_dual_20": "MGH1",
               "mgh1_dual_22": "MGH1",
               "mgh1_dual_24": "MGH1",
               "mgh1_dual_27": "MGH1",
               "mgh1_dual_29": "MGH1",
               "mgh1_dual_31": "MGH1",
               "mgh2_msc_03": "MGH2",
               "mgh2_msc_05": "MGH2",
               "mgh2_msc_10": "MGH2",
               "mgh2_msc_12": "MGH2",
               "mgh2_msc_15": "MGH2",
               "mgh2_msc_17": "MGH2",
               "mgh2_msc_19": "MGH2",
               "mgh2_msc_22": "MGH2",
               "mgh2_dual_03": "MGH2",
               "mgh2_dual_05": "MGH2",
               "mgh2_dual_06": "MGH2",
               "mgh2_dual_08": "MGH2",
               "mgh2_dual_10": "MGH2",
               "mgh2_dual_12": "MGH2",
               "mgh2_dual_13": "MGH2",
               "mgh2_dual_15": "MGH2",
               "mgh2_dual_17": "MGH2",
               "mgh2_dual_19": "MGH2",
               "mgh2_dual_20": "MGH2",
               "mgh2_dual_22": "MGH2",
               "mgh3_msc_03": "MGH3",
               "mgh3_msc_06": "MGH3",
               "mgh3_msc_08": "MGH3",
               "mgh3_msc_10": "MGH3",
               "mgh3_msc_14": "MGH3",
               "mgh3_msc_16": "MGH3",
               "mgh3_msc_19": "MGH3",
               "mgh3_msc_21": "MGH3",
               "mgh3_msc_23": "MGH3",
               "mgh3_msc_26": "MGH3",
               "mgh3_msc_28": "MGH3",
               "mgh3_msc_30": "MGH3",
               "mgh3_msc_33": "MGH3",
               "mgh3_dual_03": "MGH3",
               "mgh3_dual_06": "MGH3",
               "mgh3_dual_08": "MGH3",
               "mgh3_dual_10": "MGH3",
               "mgh3_dual_14": "MGH3",
               "mgh3_dual_16": "MGH3",
               "mgh3_dual_19": "MGH3",
               "mgh3_dual_21": "MGH3",
               "mgh3_dual_23": "MGH3",
               "mgh3_dual_26": "MGH3",
               "mgh3_dual_28": "MGH3",
               "mgh3_dual_30": "MGH3",
               "mgh3_dual_33": "MGH3",
               "uci_msc_03": "UCI",
               "uci_msc_06": "UCI",
               "uci_msc_08": "UCI",
               "uci_msc_10": "UCI",
               "uci_msc_13": "UCI",
               "uci_msc_15": "UCI",
               "uci_msc_17": "UCI",
               "uci_msc_20": "UCI",
               "uci_msc_22": "UCI",
               "uci_msc_27": "UCI",
               "uci_msc_29": "UCI",
               "uci_msc_30": "UCI",
               "uci_dual_03": "UCI",
               "uci_dual_06": "UCI",
               "uci_dual_08": "UCI",
               "uci_dual_10": "UCI",
               "uci_dual_13": "UCI",
               "uci_dual_15": "UCI",
               "uci_dual_17": "UCI",
               "uci_dual_20": "UCI",
               "uci_dual_22": "UCI",
               "uci_dual_27": "UCI",
               "uci_dual_29":"UCI",
               "uci_dual_30":"UCI",
               "ctrl_media":"Other",
               "ctrl_isc": "Other",
               "ctrl_dual":"Other",
               "ctrl_msc": "Other",
               "-": np.NaN}
    
    cytokines = hemi_totalDF.columns.values
    cytokines = cytokines[3::]

    hemi_totalDF = hemi_totalDF.replace(rename_dict)
    hemi_totalDF = hemi_totalDF.rename({"Group":"Location","Cell": "Treatment","Time (days)":"Day"},axis=1)
    cytokines = hemi_totalDF.columns.values
    cytokines = cytokines[3::]
    
    locations = hemi_totalDF["Location"].unique()
    treatments = hemi_totalDF["Treatment"].unique()
    days = np.sort(hemi_totalDF["Day"].unique())

    mean_olabisi_DF = pd.DataFrame([])
    for i, loc in enumerate(locations):
        for j, treat in enumerate(treatments):
            for k, time in enumerate(days):
                for l, mark in enumerate(cytokines):
                    conditionDF = hemi_totalDF.loc[(hemi_totalDF["Location"] == loc) & (hemi_totalDF["Treatment"] == treat) & (hemi_totalDF["Day"] == time)]
                    cytok = conditionDF[mark].values
                    
                    # Some conditions do not have any values 
                    if cytok.shape[0] == 0:
                        cytok =  np.empty(6)
                        cytok[:] = np.NaN

                    # Import made values as strings, so now need to ensure all numbers are float
                    elif type(cytok[0]) or type(cytok[1]) or type(cytok[2]) == str:
                        for m in range(len(cytok)):
                            if cytok[m] is not np.NaN:
                                cytok[m] = float(cytok[m])

                    mean = np.nanmean(cytok)
                    # Obtaining average for every condition in to a DataFrame
                    mean_olabisi_DF = pd.concat([mean_olabisi_DF, pd.DataFrame({"Location": loc, "Treatment": treat, "Day": time, "Cytokine": mark,
                                                                                "Mean":[mean]})])
                    
                    
    mean_olabisi_DF = mean_olabisi_DF.reset_index(drop=True) 
    # Shape of Tensor: Location, Treatment, Day, and Cytokine
    tensor = np.empty((len(locations), len(treatments), len(days), len(cytokines)))

    tensor[:] = np.NaN
    
    # Converting DataFrame into an Xarray
    for i, loc in enumerate(locations):
        for j, treat in enumerate(treatments):
            for k, time in enumerate(days):
                for l, mark in enumerate(cytokines):
                    entry = mean_olabisi_DF.loc[(mean_olabisi_DF["Location"] == loc) & (mean_olabisi_DF ["Treatment"] == treat) & (mean_olabisi_DF["Day"] == time) &
                                                 (mean_olabisi_DF["Cytokine"] == mark)]["Mean"].values
                    tensor[i,j,k,l] = entry
                       
    olabisiXA = xa.DataArray(tensor, dims=("Location", "Treatment", "Day", "Cytokine"), coords={"Location": locations, "Treatment": treatments,
                                            "Day": days, "Cytokine": cytokines})

    return olabisiXA 