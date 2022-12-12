# %%
import numpy as np
import pandas as pd

from inform import run_inform

# params
#soil reflectance spectrum
# df=pd.read_csv('extracted_soil_reflec.csv')
# bands=df['bands']
# soilSpec=df['reflectance']

#CANOPY
#PROSPECT
n=2        # number of plates in the leaf
ant=1      # anthocyanin content (ug.cm-2)
cm=0.015   # LMA (g.cm-2)
cw=0.015   # EWT (g.cm-2)
cab=30     # chlorophyll content (ug.cm-2)
car=10     # carotenoid content (ug.cm-2)
cbrown=0   # brown pigments (arbitrary unit)

#CANOPY
lai=4.5     # LAI of the canopy layer (m2.m-2)
typelidf=2  # method to decide on the leaf angle distribution (LAD) (1 or 2)
lidfa=50    # LAD parameters
lidfb=0
rsoil=1     # soil reflectance
psoil=1     # soil moisture parameter (0 to 1, dry to wet)
SD=902      # stem density (stem.ha-1)
CD=4.7      # crown diameter (m)
H=18        # canopy mean height (m)

#UNDERSTORY
#PROSPECT   same as for the canopy but for the understory (only matters if you dont input rsoil0 and mode_understory_reflectance==True
n_u=2
ant_u=1
cm_u=0.015
cw_u=0.015
cab_u=30
car_u=10
cbrown_u=0

#CANOPY
lai_u=2
typelidf_u=2
lidfa_u=50
lidfb_u=0
rsoil_u=1
psoil_u=1

#SUN
hspot=1     # hotspot parameter
tts=90-53.8 # sun zenith angle
tto=0.      # sensor zenith angle
phi=190.82  # relative azimuth angle between sun and sensor

#EXTRA PARAMS
model_understory_reflectance=False  # indicates if INFORM has to generate the understory reflectance or if it is given as input


R = run_inform(n, cab, car, cbrown, cw, cm, lai, tts, tto, phi, SD, CD, H, typelidf=typelidf, lidfb=lidfb, lidfa=lidfa, hspot=hspot, rsoil=rsoil, psoil=psoil, ant=ant, prot=0, cbc=0, alpha=40.0, prospect_version='D', rsoil0=np.zeros(2101),model_understory_reflectance=False, n_u=None, cab_u=None, car_u=None, cbrown_u=None, cw_u=None, cm_u=None, ant_u=None, lai_u=None, lidfa_u=None, typelidf_u=None, lidfb_u=None)
# %%
