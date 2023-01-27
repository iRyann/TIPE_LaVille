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
    return (d<200)

def distance (a,b):
    #Calcule d(a,b) en m
    lat=m.radians(a[1])
    long=m.radians(b[0])
    r=6371*(10**3)
    lat_t=m.radians(b[1])
    long_t=m.radians(b[0])
    d=2*r*m.asin(m.sqrt((m.sin((lat-lat_t)/2)**2)+m.cos(lat)*m.cos(lat_t)*(m.sin((long-long_t)/2))**2))
    return d

#---------------------------------------------------





#---------------------------------------------------
#           Récupération des données
#---------------------------------------------------

#-Base de donnée support ---------------------------
con = sql.connect('filaire_de_voirie.db')
requete2="""select geo_shape FROM "filaire-de-voirie" """


f=[] # Tableau brut des données récupérées : géopoints


#---------------------------------------------------
#           partnoeudsnement des données
#---------------------------------------------------
gd = {}
gd["latmin"]=0
gd["latmax"]=0
gd["longmin"]=0
gd["longmax"]=0


def taille_data (gd : dict, tab : list):
    if tab[0]<gd["longmin"]:
        gd["longmin"]=tab[0]
    if tab[0]>gd["longmax"]:
        gd["longmax"]=tab[0]
    if tab[1]<gd["latmin"]:
        gd["latmin"]=tab[0]
    if tab[1]>gd["latmax"]:
        gd["latmax"]=tab[0]

nlarge=np.abs((gd["latmax"]-gd["latmin"])*10**9)//2 #distance([gd["longmin"],gd["latmin"]],[gd["longmin"],gd["latmax"]])//2
nlong=np.abs((gd["longmax"]-gd["longmin"])*10**9)//2 #distance([gd["longmin"],gd["latmin"]],[gd["longmax"],gd["latmin"]])//2
partnoeuds=[]

def partition_vide(g : list, l : int , L : int ):
    for i in range(0,l):
        g.append([])
    for i in range(0,l):
        for j in range(0,L):
            g[i].append([])



#---------------------------------------------------


def recupere_data (f,con, req) :
    #Execute la requete req dans la db passé par con
    curseur = con.cursor()
    res = None
    curseur.execute(req)
    res = curseur.fetchall()
    for i in range(0,len(res)-1):
        a=eval(res[i][0])
        if est_valide(a['coordinates'][0][0]):
            f.append(a['coordinates'][0])
            for i in a['coordinates'][0]:
                taille_data(gd,i)
    return f


#On récupère les données dans f grâce à la requête requete2
t=recupere_data(f,con,requete2)




#---------------------------------------------------
#           Traitement des données
#---------------------------------------------------

g={} #Graphe
tab=[] #Coordonnées associées à un pt une graphe
noeud_valide=[]
# 1/ Création du graphe

def ajoute_voisin(g : dict, i : int , c : tuple ) :
    #Ajoute la relation i->j et j->i telle que d(i,j)=d
    (j,dist)=c
    if(not(j in g[i])):
        g[i][j]=dist
        g[j][i]=dist
    else:
        print(i,j)
        raise ValueError("Relation déjà présente")


def ajoute_noeud(g : dict, t : list, tab : list):
    n=0 #nombre de noeuds
    #len(t)-1
    for i in range(0,len(t)-1):
        for j in range(0,len(t[i])):
            g[n]={}
            tab.append(t[i][j])
            noeud_valide.append(True)
            n+=1
        #Complétion du voisinnage
        if (len(t[i]))>1:
            for j in range (0,len(t[i])-1):
                # On ajoute les relations ( j <-> j+1 ) entre les sommets intermédiaire d'une même route
                ajoute_voisin(g,n-len(t[i])+j,(n-len(t[i])+j+1,distance(t[i][j],t[i][j+1])))

#On complete le graphe g grâce aux données récupérées dans t
#en retenant les coordonnées de chaque noeud de g dans tab
ajoute_noeud(g,t,tab)

# 3/ Complétion de la partnoeuds 
partition_vide(partnoeuds,nlarge,nlong)

tabclasse=[] # Pour chaque noeud, on retient sa classe. 

def add_part(gg : dict, t: list):
    # Etant donné un graphe "gg" , on partionnent l'ensemble des noeuds selon leurs coordonées dans "t"
    for k in gg.keys():
        N=0
        E=0
        while(not(N*nlong<=tab[k][0]<=(N+1)*nlong)):
            N+=1
        while(not(E*nlarge<=tab[k][0]<=(E+1)*nlarge)):
            E+=1
        partnoeuds[N][E].append(k)
        tabclasse[k]=[N,E]

add_part(g,tab)


# 2/ Correction du graphe

## Fonctions batardes -----------
def remplace (i : int , j : int ) :
    #On ajoute les voisins de j à ceux de i
    for k,v in g[j].items():
        g[i][k]=v

    #On remplace j par i pour toutes les occurences de j
    for z,v in g.items():
        b=True
        
        r=[]
        for k in v.keys():
            if k==i:
                b=False
            
            if k==j:
                r.append(k) 
        for l in r:
            del v[l]
        del r
        if b:
            v[i]=distance(tab[z],tab[i])
        
def classe_g (g : dict):
    for i in range(0,len(g)):
            if bool(g[i]):
                for j in range(1+i,len(g)):
                    if bool(g[j]) and distance(tab[i],tab[j])<1:
                        remplace(i,j)
                        g[j]={}

# --------------------------------
    
def fusion ( i : int , j : int ):
    # Fusionne le noeud i et j, en attibuant remplaçant j par i dans les voisins de celui-ci
    for k,v in g[j].items():
        if (k!=i and not(k in g[i])):
            g[i][k]=v
        del g[k][j]
        g[k][i]=v
    noeud_valide[j]=False
    
def apptab(x : int, y : int ):
    # part[x][y] isn't out of range
    return(x>=0 and y>=0 and x<nlong and y<nlarge)


def classement(g:dict):
    ##Classe le dictionnaire g en recquérant les fusions nécessaires
    for k in g.keys:
        X=k
        temp=[]
        n=tabclasse[k][0]
        e=tabclasse[k][1]

        for i in range(-1,2):
            for j in range(-1,2):
                if apptab(n+i,e+j):
                    for z in partnoeuds[z]:
                        if z<X:
                            X=z
                        if distance(z,k)<1:
                            temp.append(z)
        for i in temp:
            fusion(X,i)
            g[i]={}


g_final={}
gf_coord=[]

def renum( t1 : list , t2 : list, g1 : dict, gf : dict):
    table=[]
    c=0
    for k in g1.keys():
        if t1[k]:
            table.append(c)
            t2.append(tab[k])
            c+=1
        else:
            table.append(-1)
    
    for k,v in g1.items():
        if table[k]!=-1:
            for kk,vv in v.items():
                if table[kk]!=-1:
                    gf[table[k]]=vv


renum(noeud_valide,gf_coord,g,g_final)





def affiche_graphe( d : dict, tab : list):
    fig, ax = plt.subplots()
    z=0
    for k,v in d.items():
        for j in v.keys():

            x=[tab[j][0],tab[k][0]]
            y=[tab[j][1],tab[k][1]]
            
            xx = np.linspace(tab[j][0],tab[k][0], 100)
            yy = ((tab[j][1]-tab[k][1])/(tab[j][0]-tab[k][0]))*(xx-tab[k][0])+tab[k][1]

           

            ax.plot(xx, yy, linewidth=0.5)

            ax.scatter(x, y, s=1.5)
            z+=1
            print(k)
            if z==1000:
                break
        if z==1000:
                break
    plt.show()

