### Convenience functions
from pathlib import Path

def translate_dir(survey, grating='prism'):
    if survey == "GS-HST-DEEP" and grating == 'prism':
        return GS_HST_DEEP, "v4.0"
    elif survey == 'GS-HST-DEEP' and grating == 'g395m':
        return GS_HST_DEEP_g395m, "v4.0"
    elif survey == 'GS-HST-DEEP' and grating == 'g140m':
        return GS_HST_DEEP_g140m, "v4.0"

    elif survey == "GS-JWST-DEEP" and grating == 'prism':
        return GS_JWST_1287, "v4.0"
    elif survey == 'GS-JWST-DEEP' and grating == 'g395m':
        return GS_JWST_1287_g395m, "v4.0"
    elif survey == 'GS-JWST-DEEP' and grating == 'g140m':
        return GS_JWST_1287_g140m, "v4.0"

    elif survey == "GS-3215" and grating == 'prism':
        return GS_3215, "v4.0"
    elif survey == 'GS-3215' and grating == 'g395m':
        return GS_3215_g395m, "v4.0"
    elif survey == 'GS-3215' and grating == 'g140m':
        return GS_3215_g140m, "v4.0"

    elif survey == "GS-HST-MEDIUM" and grating == 'prism':
        return GS_HST_MEDIUM, "v4.0"
    elif survey == 'GS-HST-MEDIUM' and grating == 'g395m':
        return GS_HST_MEDIUM_g395m, "v4.0"
    elif survey == 'GS-HST-MEDIUM' and grating == 'g140m':
        return GS_HST_MEDIUM_g140m, "v4.0"

    elif survey == "GS-JWST-MEDIUM-1180" and grating == 'prism':
        return GS_JWST_MEDIUM_1180, "v4.0"
    elif survey == 'GS-JWST-MEDIUM-1180' and grating == 'g395m':
        return GS_JWST_MEDIUM_1180_g395m, "v4.0"
    elif survey == 'GS-JWST-MEDIUM-1180' and grating == 'g140m':
        return GS_JWST_MEDIUM_1180_g140m, "v4.0"

    elif survey == "GS-JWST-MEDIUM-1286" and grating == 'prism':
        return GS_JWST_MEDIUM_1286, "v4.0"
    elif survey == 'GS-JWST-MEDIUM-1286' and grating == 'g395m':
        return GS_JWST_MEDIUM_1286_g395m, "v4.0"
    elif survey == 'GS-JWST-MEDIUM-1286' and grating == 'g140m':
        return GS_JWST_MEDIUM_1286_g140m, "v4.0"

    elif survey == "GN-HST-MEDIUM" and grating == 'prism':
        return GN_HST_MEDIUM, "v4.0"
    elif survey == 'GN-HST-MEDIUM' and grating == 'g395m':
        return GN_HST_MEDIUM_g395m, "v4.0"
    elif survey == 'GN-HST-MEDIUM' and grating == 'g140m':
        return GN_HST_MEDIUM_g140m, "v4.0"

    elif survey == "GN-JWST-MEDIUM" and grating == 'prism':
        return GN_JWST_MEDIUM, "v4.0"
    elif survey == 'GN-JWST-MEDIUM' and grating == 'g395m':
        return GN_JWST_MEDIUM_g395m, "v4.0"
    elif survey == 'GN-JWST-MEDIUM' and grating == 'g140m':
        return GN_JWST_MEDIUM_g140m, "v4.0"

def translate_ID(ID):
    if ID < 1000:
        return f"000{ID}"
    elif ID < 10000:
        return f"00{ID}"
    elif ID < 100000:
        return f"0{ID}"
    else:
        return f"{ID}"



### DIRECTORIES
### DEEP pointings
GS_HST_DEEP = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-deephst/Final_products_v4.0_GTO/prism_clear/"
GS_HST_DEEP_g395m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-deephst/Final_products_v4.0_GTO/g395m_f290lp/"
GS_HST_DEEP_g140m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-deephst/Final_products_v4.0_GTO/g140m_f070lp/"

GS_3215 = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-ultradeep/Final_products_v4.0_GTO/prism_clear/"
GS_3215_g395m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-ultradeep/Final_products_v4.0_GTO/g395m_f290lp/"
GS_3215_g140m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-ultradeep/Final_products_v4.0_GTO/g140m_f070lp/"

GS_JWST_1287 = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-deepjwst/Final_products_v4.0_GTO/prism_clear/"
GS_JWST_1287_g395m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-deepjwst/Final_products_v4.0_GTO/g395m_f290lp/"
GS_JWST_1287_g140m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-deepjwst/Final_products_v4.0_GTO/g140m_f070lp/"

# GS_HST_DEEP = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-HST-DEEP/Final_products_v4.0/prism_clear/"
# GS_HST_DEEP_g395m = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-HST-DEEP/Final_products_v3.0/g395m_f290lp/"
# GS_HST_DEEP_g140m = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-HST-DEEP/Final_products_v3.0/g140m_f070lp/"
# hst_deep_table = pd.read_csv("/Users/aayushsaxena/Desktop/Oxford/JADES/GS-HST-DEEP/redshifts/visual_redshifts_gs_deep_hst_fde_v3.0_v1.0.csv")

# GS_3215 = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-3215/Final_products_v3.1/prism_clear/"
# GS_3215_g395m = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-3215/Final_products_v3.1/g395m_f290lp/"
# gs_deep_3215_table = pd.read_csv("/Users/aayushsaxena/Desktop/Oxford/JADES/GS-3215/visual_redshifts_consolidated_ultra_deep_gs_3215.csv")

# GS_JWST_1287 = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-1287/Final_products_v3.1/prism_clear/"
# GS_JWST_1287_g395m = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-1287/Final_products_v3.1/g395m_f290lp/"
# gs_jwst_deep_table = pd.read_csv("/Users/aayushsaxena/Desktop/Oxford/JADES/GS-1287/visual_redshifts_consolidated_1287.csv")

### MEDIUM pointings
GS_HST_MEDIUM = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-mediumhst/Final_products_v4.0_GTO/prism_clear/"
GS_HST_MEDIUM_g395m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-mediumhst/Final_products_v4.0_GTO/g395m_f290lp/"
GS_HST_MEDIUM_g140m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-mediumhst/Final_products_v4.0_GTO/g140m_f070lp/"

GS_JWST_MEDIUM_1180 = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-mediumjwst1180/Final_products_v4.0_GTO/prism_clear/"
GS_JWST_MEDIUM_1180_g395m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-mediumjwst1180/Final_products_v4.0_GTO/g395m_f290lp/"
GS_JWST_MEDIUM_1180_g140m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-mediumjwst1180/Final_products_v4.0_GTO/g140m_f070lp/"


GS_JWST_MEDIUM_1286 = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-mediumjwst/Final_products_v4.0_GTO/prism_clear/"
GS_JWST_MEDIUM_1286_g395m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-mediumjwst/Final_products_v4.0_GTO/g395m_f290lp/"
GS_JWST_MEDIUM_1286_g140m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-s-mediumjwst/Final_products_v4.0_GTO/g140m_f070lp/"

GN_HST_MEDIUM = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-n-mediumhst/Final_products_v4.0_GTO/prism_clear/"
GN_HST_MEDIUM_g395m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-n-mediumhst/Final_products_v4.0_GTO/g395m_f290lp/"
GN_HST_MEDIUM_g140m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-n-mediumhst/Final_products_v4.0_GTO/g140m_f070lp/"

GN_JWST_MEDIUM = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-n-mediumjwst/Final_products_v4.0_GTO/prism_clear/"
GN_JWST_MEDIUM_g395m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-n-mediumjwst/Final_products_v4.0_GTO/g395m_f290lp/"
GN_JWST_MEDIUM_g140m = "/Volumes/data5tb/JADES-NIRSpec/v4.0/goods-n-mediumjwst/Final_products_v4.0_GTO/g140m_f070lp/"

# GS_HST_MEDIUM = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-HST-MEDIUM/Final_products_v3.1/prism_clear/"
# GS_HST_MEDIUM_g395m = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-HST-MEDIUM/Final_products_v3.1/g395m_f290lp/"
# gs_hst_medium_table = pd.read_csv("/Users/aayushsaxena/Desktop/Oxford/JADES/GS-HST-MEDIUM/visual_redshifts_hst_medium_redo_v0.2.csv")

# GS_JWST_MEDIUM_1180 = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-JWST-MEDIUM-1180/Final_products_v3.1/prism_clear/"
# GS_JWST_MEDIUM_1180_g395m = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-JWST-MEDIUM-1180/Final_products_v3.1/g395m_f290lp/"
# gs_jwst_medium_1180_table = pd.read_csv("/Users/aayushsaxena/Desktop/Oxford/JADES/GS-JWST-MEDIUM-1180/visual_redshifts_JWST_Medium_1180_v0.2.csv")

# GS_JWST_MEDIUM_1286 = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-JWST-MEDIUM-1286/Final_products_v3.1/prism_clear/"
# GS_JWST_MEDIUM_1286_g395m = "/Users/aayushsaxena/Desktop/Oxford/JADES/GS-JWST-MEDIUM-1286/Final_products_v3.1/g395m_f290lp/"
# gs_jwst_medium_1286_table = pd.read_csv("/Users/aayushsaxena/Desktop/Oxford/JADES/GS-JWST-MEDIUM-1286/visual_redshifts_JWST_Medium_1282_Obs1_20230908.csv")

# GN_HST_MEDIUM = "/Users/aayushsaxena/Desktop/Oxford/JADES/GN-HST-MEDIUM/Final_products_v3.1/prism_clear/"
# GN_HST_MEDIUM_g395m = "/Users/aayushsaxena/Desktop/Oxford/JADES/GN-HST-MEDIUM/Final_products_v3.1/g395m_f290lp/"
# gn_hst_medium_table = pd.read_csv("/Users/aayushsaxena/Desktop/Oxford/JADES/GN-HST-MEDIUM/visual_redshifts_consolidated_gn_hst_medium_1181_20230830A.csv")

# GN_JWST_MEDIUM = "/Users/aayushsaxena/Desktop/Oxford/JADES/GN-JWST-MEDIUM/Final_products_v3.1/prism_clear/"
# GN_JWST_MEDIUM_g395m = "/Users/aayushsaxena/Desktop/Oxford/JADES/GN-JWST-MEDIUM/Final_products_v3.1/g395m_f290lp/"
# gn_jwst_medium_table = pd.read_csv("/Users/aayushsaxena/Desktop/Oxford/JADES/GN-JWST-MEDIUM/visual_redshifts_consolidated_gn_jwst_medium_1181.csv")



def select_sources(input_cat, specdir, redshift_range, grating, version):
    speclist = []
    redshifts = []
    z_low = redshift_range[0]
    z_high = redshift_range[1]

    if grating == 'prism':
        file_version = 'prism_clear'
    elif grating == 'g395m':
        file_version = 'g395m_f290lp'
    elif grating == 'g140m':
        file_version = 'g140m_f070lp'

    for i in range(len(input_cat)):
        if input_cat["z_visinsp"][i] > z_low and input_cat["z_visinsp"][i] < z_high:
            specID = translate_ID(input_cat["NIRSpec_ID"][i])
            file_name = f"{specID}_{file_version}_{version}_1D.fits"
            file_path = Path(specdir+f"{file_name}")
            if file_path.exists():
                speclist.append(specdir+f"{file_name}")
                redshifts.append(input_cat["z_visinsp"][i])
            else:
                print(f"Warning: {file_path} not found. Skipping...")

    return(speclist, redshifts)
