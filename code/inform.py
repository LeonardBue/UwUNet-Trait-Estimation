import prosail
import numpy as np
import pandas as pd

#soil reflectance spectrum
# df = pd.read_csv('extracted_soil_reflec.csv')
# bands =  df['bands']
# soilSpec = df['reflectance']


def run_inform(n, cab, car, cbrown, cw, cm, lai, tts, tto, phi, SD, CD, H, typelidf=2, lidfb=0, lidfa=50, hspot=1, rsoil=1, psoil=1, ant=0, prot=0, cbc=0, alpha=40.0, prospect_version='PRO', rsoil0=np.zeros(2101), model_understory_reflectance=False, n_u=None, cab_u=None, car_u=None, cbrown_u=None, cw_u=None, cm_u=None, ant_u=None, lai_u=None, lidfa_u=None, typelidf_u=None, lidfb_u=None):
    """
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
    model_understory_reflectance=False # indicates if INFORM has to generate the understory reflectance or if it is given as input
    """


    # RUN INFORM
    tss,too,tsstoo,rdd,tdd,rsd,tsd,rdo,tdo,rso,rsos,rsod,rddt,rsdt,rdot,rsodt,rsost,rsot,gammasdf,gammasdb,gammaso = prosail.run_prosail(n, cab, car, cbrown, cw, cm, lai, lidfa, hspot, tts, tto, phi, ant=ant, prot=0, cbc=cbc, alpha=40.0, prospect_version='PRO', typelidf=typelidf, lidfb=lidfb, rsoil0=rsoil0,factor='ALLALL')
    
    laiinf=20
    Rc = prosail.run_prosail(n, cab, car, cbrown, cw, cm, laiinf, lidfa, hspot, tts, tto, phi, ant=ant, prot=prot,cbc=cbc, alpha=alpha, prospect_version=prospect_version, typelidf=typelidf, lidfb=lidfb, rsoil0=rsoil0)
    
    if model_understory_reflectance==False:
        Rg=rsoil0
    else:
        Rg = prosail.run_prosail(n_u, cab_u, car_u, cbrown_u, cw_u, cm_u, lai_u, lidfa_u, hspot, tts, tto, phi, ant=ant_u, prot=prot,cbc=cbc, alpha=alpha, prospect_version=prospect_version, typelidf=typelidf_u, lidfb=lidfb_u, rsoil0=rsoil0)
    
    Ts=tss
    To=too
    tts=tts*np.pi/180
    tto=tto*np.pi/180
    phi=phi*np.pi/180
    
    k=np.pi*(0.5*CD)**2/1e4
    cs=1-np.exp((-k*SD)/np.cos(tts))
    co=1-np.exp((-k*SD)/np.cos(tto))
    g=(np.tan(tts)**2+np.tan(tto)**2-2*np.tan(tts)*np.tan(tto)*np.cos(phi))**0.5 
    corr=np.exp(-g*H/CD)
    Fa=co*cs+corr*(co*(1-co)*cs*(1-cs))**0.5
    Fb=co*(1-cs)-corr*(co*(1-co)*cs*(1-cs))**0.5
    Fc=cs*(1-co)-corr*(co*(1-co)*cs*(1-cs))**0.5
    Fd=(1-co)*(1-cs)+corr*(co*(1-co)*cs*(1-cs))**0.5
    G=Fa*Ts*To+Fb*To+Fc*Ts+Fd
    C=(1-Ts*To)*cs*co
    R=Rc*C+Rg*G

    return R

