import sqlite3 as sql
import matplotlib.pyplot as plt
import math as m 
import numpy as np
#---------------------------------------------------
#           Fonctions auxiliaires
#---------------------------------------------------

def est_valide (tab):
    #Retourne si dist(tab,capitole)<10 Km où tab=[long,lat]
    lat=m.radians(tab[1])
    long=m.radians(tab[0])
    r=6371*(10**3)
    lat_t=m.radians(43.60395967066511)
    long_t=m.radians(1.4433469299842416)
    d=2*r*m.asin(m.sqrt((m.sin((lat-lat_t)/2)**2)+m.cos(lat)*m.cos(lat_t)*(m.sin((long-long_t)/2))**2))
    return (d<500)

def distance (a,b):
    #Calcule d(a,b) en m
    lat=m.radians(a[1])
    long=m.radians(a[0])
    r=6371*(10**3)
    lat_t=m.radians(b[1])
    long_t=m.radians(b[0])
    d=2*r*m.asin(m.sqrt((m.sin((lat-lat_t)/2)**2)+m.cos(lat)*m.cos(lat_t)*(m.sin((long-long_t)/2))**2))
    return d



def affiche_graphe( d : dict, tab : list):
    fig, ax = plt.subplots()
    #----- Affichage de la grille ----------------------
    grid_x_ticks = np.arange(gd["latmin"],gd["latmax"], 0.000001)
    grid_y_ticks = np.arange(gd["longmin"], gd["longmin"], 0.000001)

    ax.set_xticks(grid_x_ticks , minor=True)
    ax.set_yticks(grid_y_ticks , minor=True)
    print("coucou")
    ax.grid(which='both')

    ax.grid(which='minor', alpha=0.2, linestyle='--')
    #----------------------------------------------------
    z=0
    a=43.60395967066511
    b=1.4433469299842416
    ax.scatter(b,a,s=5)
    for k,v in d.items():
        for j in v.keys():

            x=[tab[j][0],tab[k][0]]
            y=[tab[j][1],tab[k][1]]
            e=tab[j][0]-tab[k][0]
            if e==0:
                e=10**(-16)


            xx = np.linspace(tab[j][0],tab[k][0], 100)
            yy = ((tab[j][1]-tab[k][1])/(e))*(xx-tab[k][0])+tab[k][1]

            ax.plot(xx, yy, linewidth=0.5)

            ax.scatter(x, y, s=2.5)
            z+=1
    plt.plot(abs,ord,'g^')        
    plt.show()




#4/ Traitement du graphe 

## --------- Ajout de la population ------------ #



q=[] # Partitionnement des noeuds par quartier

partition_vide(q,,1)