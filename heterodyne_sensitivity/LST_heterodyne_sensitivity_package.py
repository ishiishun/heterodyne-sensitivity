#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io, requests
from astropy import units as u
from jupyter_io import savetable_in_notebook
import itables
import itables.options as opt
itables.init_notebook_mode(all_interactive=True)
# type aliases
ArrayLike = Union[np.ndarray, List[float], List[int], float, int]
# others 
c = 2.99792458e8 #m/s
EL = 60. #deg.

# In[3]:

def load_Tsys():
    url_Tsys="https://github.com/ishiishun/lst/raw/master/data/Tsys.csv"
    s_Tsys=requests.get(url_Tsys).content
    df_Tsyss = pd.read_csv(io.StringIO(s_Tsys.decode('utf-8')),
            header=0,
            index_col=0
        )
    return df_Tsyss


# In[4]:


def PlotTsys(df_Tsyss):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    df_Tsyss['Tsys'][84.0:850].plot(ax = ax)
    #x = df_ratio.index
    #ax.plot(x, x/x*np.sqrt(2), "-", linewidth=1,color="b")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("System noise temperature, Tsys (K)")
    ax.set_yscale("log")
    #ax.set_xlim(Bands[band]["f_min"], Bands[band]["f_max"])
    ax.set_xlim(84, 850)
    ax.set_ylim(10, np.min([df_Tsyss["Tsys"].max(), 10000.]))
    ax.tick_params(direction="in", which="both")
    ax.grid(True)
    ax.set_title(
        "EL = {} deg".format(EL),
        fontsize=12,
    )
    ax.legend()
    fig.tight_layout()

    # Create download link
    F = df_Tsyss.index
    df_download = pd.DataFrame(data=F, columns=["F(GHz)"])
    df_download["Tsys(K)"] = df_Tsyss['Tsys']

    return savetable_in_notebook(df_download, "Tsys.csv")
    


# In[5]:


def LST_HPBW(F: ArrayLike, D = 50.) -> ArrayLike:
    
    wavelength =  c/F
    return 1.21 * wavelength/D

def PlotHPBW_LST():
    """Plot half power beam width of LST.

    Returns
    -------
    html
        HTML object for download link (to be used in Jupyter notebook).

    """
    F = np.arange(84., 850.1, 0.1)*1e9 

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(
        F / 1e9,
        LST_HPBW(F) * 180 * 60 * 60 / np.pi,
        linewidth=1,
        color="b",
        alpha=1,
        label="D=50 m",
    )

    ax.plot(
        F / 1e9,
        LST_HPBW(F, 30) * 180 * 60 * 60 / np.pi,
        linewidth=1,
        color="g",
        alpha=1,
        label="D=30 m (inner)",
    )

    ax.axvline(x = 420., color = 'gray', linestyle = "--")
    ax.text(421, 15, "420 GHz", rotation=90, verticalalignment='center')
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("HPBW (arcsec)")
    ax.set_yscale("linear")
    ax.set_xlim(84, 850)
    ax.set_ylim(0, 20)
    ax.tick_params(direction="in", which="both")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    # Create download link
    df_download = pd.DataFrame(data=F, columns=["F(GHz)"])
    df_download["HPBW_50m(arcsec)"] = LST_HPBW(F) * 180 * 60 * 60 / np.pi
    df_download["HPBW_30m(arcsec)"] = LST_HPBW(F, 30.) * 180 * 60 * 60 / np.pi

    return savetable_in_notebook(df_download, "HPBW.csv")


# In[6]:


def get_dTa_50m(F,                      # frequency, in Hz
            HPBW,                   # HPBW, in arcsec.          
            df_Tsyss,               # Tsys dataframe
            dV,               # velocity resolution in km/s
            N_beam,          # number of beams
            dTa_des,          # desired sensitivity in K
            sample_length = 1./5,   # length of the scan per a sample of integration, in HPBW 
            cell_size = 1./9,       # cell size, in HPBW
            sep_scan = 1./3,        # separation of raster scans, in HPBW
            t_dump = 0.1,           # in sec.
            t_sub_on = 20,          # in sec.
            N_scan = 2,             # number of the raster scans per an OFF scan
            d_OFF = 60.,            # separation between ON and OFF, in arcmin., assuming a 1-deg separation
            n_pol = 2.              # number of pol
           ):
    F = F/1e9
    v = HPBW*sample_length/t_dump   # scan velociy 
    cell = HPBW*cell_size           # cell size (Brogan & Hunter 2014)
    sep_scan = HPBW*sep_scan  
    w_x = v*t_sub_on
    w_y = w_x
    area = w_x*w_y/3600./3600.      # in deg^2
    N_row = np.ceil(w_y/sep_scan)
    t_os = t_sub_on*N_row
    t_tran = 7.
    t_app = 6.
    t_tran_OFF = np.ceil(4.4*d_OFF**(0.26)) 
    t_OH = 2*t_tran_OFF/N_scan + t_app + t_tran*(N_scan-1)/N_scan
    f_cal = 16./15. # R-Sky every 15 min.
    eta = 10.2
    t_sub_off = np.sqrt((t_sub_on + t_OH)* eta*cell*t_sub_on/w_x)*np.sqrt(N_scan)
    t_cell_on = eta*t_sub_on*cell*cell/(w_x*sep_scan)*N_beam*n_pol
    t_cell_off = t_sub_off*cell/sep_scan*N_beam*n_pol
    dfreq = F*dV/(c/1.e3)*1.e9
    dTa = df_Tsyss['Tsys'][F[0]:F[-1]].values*np.sqrt(1/t_cell_on + 1/t_cell_off)/np.sqrt(dfreq)
    Tsys_factor_nro = 1.1*np.exp(-0.18*(1/np.sin(70./180*np.pi)-1/np.sin(EL/180*np.pi)))
    dTa_nro = Tsys_factor_nro*dTa/0.88 #devided by the efficiency of the spectrometer
    t_total = N_row*(t_sub_on + t_OH + t_sub_off/N_scan)*f_cal
    df_dTa = pd.DataFrame(index = F)
    df_dTa.index.name = "F"
    df_dTa["Tsys_50m"] = df_Tsyss['Tsys'][F[0]:F[-1]].values
    df_dTa["HPBW_50m"] = HPBW
    df_dTa["dTa_50m"] = dTa
    df_dTa["t_total_50m"] = t_total/60./60.
    df_dTa["map_size_50m"] = w_x
    df_dTa["area_50m"] = area
    t_total_for_dTa_des_1deg2 = (t_total/60./60.)*np.power(dTa/dTa_des, 2.)/area
    df_dTa["t_total_dTa_des_1deg2_50m"] = t_total_for_dTa_des_1deg2
    return df_dTa, N_beam, t_dump, dTa_des


# In[7]:


def get_dTa_30m(F,                      # frequency, in Hz
            HPBW,                   # HPBW, in arcsec.          
            df_Tsyss,               # Tsys dataframe
            dV,               # velocity resolution in km/s
            N_beam,          # number of beams
            dTa_des,          # desired sensitivity in K
            sample_length = 1./5,   # length of the scan per a sample of integration, in HPBW 
            cell_size = 1./9,       # cell size, in HPBW
            sep_scan = 1./3,        # separation of raster scans, in HPBW
            t_dump = 0.1,           # in sec.
            t_sub_on = 20,          # in sec.
            N_scan = 2,             # number of the raster scans per an OFF scan
            d_OFF = 60.,            # separation between ON and OFF, in arcmin., assuming a 1-deg separation
            n_pol = 2.              # number of pol
           ):
    F = F/1e9
    v = HPBW*sample_length/t_dump   # scan velociy 
    cell = HPBW*cell_size           # cell size (Brogan & Hunter 2014)
    sep_scan = HPBW*sep_scan  
    w_x = v*t_sub_on
    w_y = w_x
    area = w_x*w_y/3600./3600.      # in deg^2
    N_row = np.ceil(w_y/sep_scan)
    t_os = t_sub_on*N_row
    t_tran = 7.
    t_app = 6.
    t_tran_OFF = np.ceil(4.4*d_OFF**(0.26)) 
    t_OH = 2*t_tran_OFF/N_scan + t_app + t_tran*(N_scan-1)/N_scan
    f_cal = 16./15. # R-Sky every 15 min.
    eta = 10.2
    t_sub_off = np.sqrt((t_sub_on + t_OH)* eta*cell*t_sub_on/w_x)*np.sqrt(N_scan)
    t_cell_on = eta*t_sub_on*cell*cell/(w_x*sep_scan)*N_beam*n_pol
    t_cell_off = t_sub_off*cell/sep_scan*N_beam*n_pol
    dfreq = F*dV/(c/1.e3)*1.e9
    dTa = df_Tsyss['Tsys'][F[0]:F[-1]].values*np.sqrt(1/t_cell_on + 1/t_cell_off)/np.sqrt(dfreq)
    Tsys_factor_nro = 1.1*np.exp(-0.18*(1/np.sin(70./180*np.pi)-1/np.sin(EL/180*np.pi)))
    dTa_nro = Tsys_factor_nro*dTa/0.88 #devided by the efficiency of the spectrometer
    t_total = N_row*(t_sub_on + t_OH + t_sub_off/N_scan)*f_cal
    df_dTa = pd.DataFrame(index = F)
    df_dTa.index.name = "F"
    df_dTa["Tsys_30m"] = df_Tsyss['Tsys'][F[0]:F[-1]].values
    df_dTa["HPBW_30m"] = HPBW
    df_dTa["dTa_30m"] = dTa
    df_dTa["t_total_30m"] = t_total/60./60.
    df_dTa["map_size_30m"] = w_x
    df_dTa["area_30m"] = area
    t_total_for_dTa_des_1deg2 = (t_total/60./60.)*np.power(dTa/dTa_des, 2.)/area
    df_dTa["t_total_dTa_des_1deg2_30m"] = t_total_for_dTa_des_1deg2
    return df_dTa, N_beam, t_dump, dTa_des


# In[8]:


def get_dTa(df_Tsyss, dV, N_beam, dTa_des, D1=50., D2= 30.):
    F = df_Tsyss.index*1.e9
    HPBW_50m = LST_HPBW(F, D=D1) * 180 * 60 * 60 / np.pi
    HPBW_30m = LST_HPBW(F, D=D2) * 180 * 60 * 60 / np.pi
    df_dTa_50m, N_beam_50m, t_dump_50m, dTa_des_50m = get_dTa_50m(F, HPBW_50m, df_Tsyss, dV = dV, N_beam = N_beam, dTa_des = dTa_des)
    df_dTa_30m, N_beam_30m, t_dump_30m, dTa_des_30m = get_dTa_30m(F, HPBW_30m, df_Tsyss, dV = dV, N_beam = N_beam, dTa_des = dTa_des)
    df_dTa = pd.merge(df_dTa_50m, df_dTa_30m, left_index=True, right_index=True)
    
    return df_dTa, N_beam_50m, t_dump_50m, dTa_des_50m

    


# In[9]:


def plot_Sensitivity(df_dTa):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(df_dTa.index, df_dTa["dTa_50m"], label = "LST")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("1-sigma sensitivity, dTa* (K)")
    ax.set_ylim(0, 0.4)
    ax.tick_params(direction="in", which="both")
    ax.grid(True)
    ax.set_title(
        "EL = {} deg, N_beam = {}, t_dump = {} sec., Dual polarization".format(EL, int(N_beam), t_dump),
        fontsize=12,
    )
    ax.legend()
    fig.tight_layout()


# In[10]:


def PlotSurveySpeed(df_dTa, N_beam, dTa_des, dV, df_lines):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot(df_dTa.index, df_dTa["t_total_dTa_des_1deg2_50m"], label = "LST")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Survey speed: time for a 1 deg^2 mapping (hr)")
    ax.set_xlim(80, 400)
    ax.set_ylim(0, 30)
    ax.tick_params(direction="in", which="both")
    ax.grid(True)
    ax.set_title(
        "EL = {} deg, N_beam = {}, dTa_desired = {} K, dV = {} km/s, area = {} deg^2, Dual polarization".format(EL, int(N_beam), dTa_des, dV, 1),
        fontsize=12,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)
    for i in df_lines.index:
        line = i
        freq = df_lines["Frequency"][i]
        if (freq > 80 and 400 > freq):
            ax.axvline(x = freq, color = 'gray', linestyle = "--", label = 'axvline - full height')
            ax.text(freq+1, 15, line, rotation=90, verticalalignment='center')

    fig.tight_layout()
    plt.show()
    


# In[11]:


def PlotSurveySpeed2(df_dTa, N_beam, dTa_des, dV, df_lines):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot(df_dTa.index, df_dTa["t_total_dTa_des_1deg2_50m"], color='blue', label = "D=50m")
    ax.plot(df_dTa.index, df_dTa["t_total_dTa_des_1deg2_30m"], color='green', label = "D=30m")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Survey speed: time for a 1 deg^2 mapping (hr)")
    ax.set_xlim(350, 500)
    ax.set_ylim(10, 5000)
    ax.set_yscale("log")
    ax.tick_params(direction="in", which="both")
    ax.grid(True)
    ax.set_title(
        "EL = {} deg, N_beam = {}, dTa_desired = {} K, dV = {} km/s, area = {} deg^2, Dual polarization".format(EL, int(N_beam), dTa_des, dV, 1),
        fontsize=12,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)
    for i in df_lines.index:
        line = i
        freq = df_lines["Frequency"][i]
        if (freq > 350 and 500 > freq):
            ax.axvline(x = freq, color = 'gray', linestyle = "--", label = 'axvline - full height')
            ax.text(freq+1, 15, line, rotation=90, verticalalignment='center')

    fig.tight_layout()
    plt.show()


# In[12]:


def PlotSurveySpeedHigh(df_dTa, N_beam, dTa_des, dV, df_lines):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot(df_dTa.index, df_dTa["t_total_dTa_des_1deg2_50m"], color='blue', label = "D=50m")
    ax.plot(df_dTa.index, df_dTa["t_total_dTa_des_1deg2_30m"], color='green', label = "D=30m")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Survey speed: time for a 1 deg^2 mapping (hr)")
    ax.set_xlim(600, 850)
    ax.set_ylim(100, 10000)
    ax.set_yscale('log')
    ax.tick_params(direction="in", which="both")
    ax.grid(True)
    ax.set_title(
        "EL = {} deg, N_beam = {}, dTa_desired = {} K, dV = {} km/s, area = {} deg^2, Dual polarization".format(EL, int(N_beam), dTa_des, dV, 1),
        fontsize=12,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)
    for i in df_lines.index:
        line = i
        freq = df_lines["Frequency"][i]
        if (freq > 600):
            ax.axvline(x = freq, color = 'gray', linestyle = "--", label = 'axvline - full height')
            ax.text(freq+1, 150, line, rotation=90, verticalalignment='center')

    fig.tight_layout()
    plt.show()


# In[13]:


def make_df_lines_low(lines, df_dTa):
    df_lines = pd.DataFrame.from_dict(lines, orient='index').rename(columns={0:'Frequency'})
    # df_lines2 = pd.DataFrame.from_dict(lines2, orient='index').rename(columns={0:'Frequency'})
    df2 = pd.DataFrame()
    for i in df_lines.index:
        freq = df_lines["Frequency"][i]
        freq = np.round(freq,1)
        s = df_dTa[freq:freq]
        if (len(s) == 0):
            print("%s at %f is out of frequency coverage due to low atmospheric transmission" % (i, df_lines["Frequency"][i]))
            df_lines.drop(i,axis=0,inplace = True)
        else:
            df2 = df2.append(df_dTa[freq:freq])
    df_lines["HPBW_50m(arcsec)"] = df2["HPBW_50m"].values
    df_lines["HPBW_30m(arcsec)"] = df2["HPBW_30m"].values
    df_lines["Tsys"] = df2["Tsys_50m"].values
    #df_lines["dTa"] = df2["dTa"].values
    #df_lines["t_total"] = df2["t_total"].values
    #df_lines["map_size"] = df2["map_size"].values
    #df_lines["area"] = df2["area"].values
    df_lines["survey_speed_50m(hr)"] = df2["t_total_dTa_des_1deg2_50m"].values
    df_lines["survey_speed_30m(hr)"] = df2["t_total_dTa_des_1deg2_30m"].values
    
    return df_lines

def export_list(df_lines):
    pd.options.display.precision = 3
    opt.lengthMenu = [100]
    itables.show(df_lines)
    return savetable_in_notebook(df_lines, "MappingSpeed.csv")



