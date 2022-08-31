from contextvars import copy_context
import string
import xarray as xa
import csv
import numpy as np
import seaborn as sns
import pandas as pd

def import_olabisi_hemi_xa():
    """"Import of the Olabisi cytokine data of aggregated dataset"""
    # str = unicode(str, errors='replace')
    hemi_totalDF = pd.read_csv("olabisi/olabisi_hemi_edited.csv").reset_index(drop=True)
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
    cytok_none = ["sCD40L","hVEGF-A"]
    
    for cytoknone in cytok_none:
        hemi_totalDF[cytoknone] = hemi_totalDF[cytoknone].fillna(np.NaN, inplace=True)
        cytokvalues = hemi_totalDF[cytoknone].values
        for i in range(len(cytokvalues)):
            if cytokvalues[i] is None:
                cytokvalues[i] = np.NaN
        hemi_totalDF[cytoknone] = cytokvalues
    

    print()
    # hemi_totalDF = hemi_totalDF.dropna(axis=1,how="all")
    
    hemi_totalDF = hemi_totalDF.replace(rename_dict)
    hemi_totalDF = hemi_totalDF.rename({"Group":"Location","Cell": "Treatment","Time (days)":"Day"},axis=1)
    cytokines = hemi_totalDF.columns.values
    cytokines = cytokines[3::]
    
    # print()
    hemi_totalDF.sort_values(by=["Location", "Treatment", "Day"], inplace=True)
    # print(cytokines)
    # print(hemi_totalDF)
    
    # print(hemi_totalDF["Location"].unique())
    # print(hemi_totalDF["Treatment"].unique())
    # print(hemi_totalDF["Day"].unique())
    

    mean_olabisi_DF = pd.DataFrame([])
    for i, loc in enumerate(hemi_totalDF["Location"].unique()):
        for j, treat in enumerate(hemi_totalDF["Treatment"].unique()):
            for k, time in enumerate(hemi_totalDF["Day"].values):
                for l, mark in enumerate(cytokines):
                    conditionDF = hemi_totalDF.loc[(hemi_totalDF["Location"] == loc) & (hemi_totalDF["Treatment"] == treat) & (hemi_totalDF["Day"] == time)]
                    cytok = conditionDF[mark].values
                    print(cytok)
                    print(loc,treat,time,mark)
                    print(type(cytok))
                    print(np.shape(cytok))
                    if cytok.shape[0] == 0:
                        cytok =  np.empty(6)
                        cytok[:] = np.NaN

                    elif type(cytok[0]) or type(cytok[1]) or type(cytok[2]) == str:
                        print("yes")
                        for m in range(len(cytok)):
                            if cytok[m] is not np.NaN:
                                cytok[m] = float(cytok[m])
                         
                    # if np.all(np.isnan(cytok)) == True:
                    #     print("yikes")
                    
                  
                    
                    # print(cytok)
                    mean = np.nanmean(conditionDF[mark].values)
                    mean_olabisi_DF = pd.concat([mean_olabisi_DF, pd.DataFrame({"Location": loc, "Treatment": treat, "Day": time, mark:[mean]})])
                    
                    
                    
                    
                    
    print(mean_olabisi_DF)          
    
    # hemi_XA = hemi_totalDF.set_index(["Location", "Treatment", "Day"]).to_xarray()
    # flowDF = flowDF[cytokines].to_array(dim="Cytokines")

    return hemi_totalDF