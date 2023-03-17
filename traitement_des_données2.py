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
    return (d<1000)

def distance (a,b):
    #Calcule d(a,b) en m
    lat=m.radians(a[1])
    long=m.radians(a[0])
    r=6371*(10**3)
    lat_t=m.radians(b[1])
    long_t=m.radians(b[0])
    d=2*r*m.asin(m.sqrt((m.sin((lat-lat_t)/2)**2)+m.cos(lat)*m.cos(lat_t)*(m.sin((long-long_t)/2))**2))
    return d

#---------------------------------------------------
#   Fonction d'affichage
#---------------------------------------------------

def affiche_graphe( d : dict, tab : list):
    fig, ax = plt.subplots()
    #----- Affichage de la grille ----------------------
    
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

            print(z)
            z+=1
    plt.plot(abs,ord,'g^')        
    plt.show()





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

gd["longmin"]=1.4433469299842416
gd["longmax"]=1.4433469299842416
gd["latmin"]=43.60395967066511
gd["latmax"]=43.60395967066511



def taille_data (gd : dict, tab : list):
    if tab[0]<gd["longmin"]:
        gd["longmin"]=tab[0]
    if tab[0]>gd["longmax"]:
        gd["longmax"]=tab[0]
    if tab[1]<gd["latmin"]:
        gd["latmin"]=tab[1]
    if tab[1]>gd["latmax"]:
        gd["latmax"]=tab[1]



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


partnoeuds=[]

#On récupère les données dans f grâce à la requête requete2
t=recupere_data(f,con,requete2)




#---------------------------------------------------
#           Traitement des données
#---------------------------------------------------


# 1/ Création du graphe

g={} #Graphe dict de dict 
tab=[] #Coordonnées associées à un pt une graphe
noeud_valide=[] # Un noeud est dit valide (ie à "true") ssi il est à considérer après fusion du graphe 

def ajoute_voisin(g : dict, i : int , c : tuple ) :
    #Ajoute la relation i->j et j->i telle que d(i,j)=dist
    (j,dist)=c
    if(not(j in g[i])):
        g[i][j]=dist
        g[j][i]=dist
    else:
        print(i,j)
        raise ValueError("Relation déjà présente")

abs=[]
ord=[]

def ajoute_noeud(g : dict, t : list, tab : list):
    n=0 #nombre de noeuds
    
    for i in range(0,len(t)):
        
        for j in range(0,len(t[i])):
            g[n]={}
            tab.append(t[i][j])
            abs.append(t[i][j][0])
            ord.append(t[i][j][1])
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


# 2/ Complétion de la partnoeuds 

##Calcul distance partition
eta = 3 #distance de partionnement
nlat=int(distance([gd["longmin"],gd["latmin"]],[gd["longmin"],gd["latmax"]])/eta)
nlong=int(distance([gd["longmin"],gd["latmin"]],[gd["longmax"],gd["latmin"]])/eta)


partition_vide(partnoeuds,nlat,nlong) # On crée la partition vide 

tabclasse=[] # Pour chaque noeud, on retient sa classe. 


def add_part(gg : dict, t: list):
    # Etant donné un graphe "gg" , on partionnent l'ensemble des noeuds selon leurs coordonées dans "t"
    
    
    for k in gg.keys():
        
        N=0
        E=0
        dlong=distance([gd["longmin"],tab[k][1]],tab[k])
        dlat=distance([tab[k][0],gd["latmin"]],tab[k])
        
        
        while(not(N*nlong<=dlong and dlong<=(N+1)*nlong)):
            N+=1
            
           
            
        while(not(E*nlat<=dlat and dlat<=(E+1)*nlat)):
            E+=1
            
        partnoeuds[N][E].append(k) #On ajoute le noeud dans la partition
        
        tabclasse.append([N,E]) #On retient la classe du noeud 
    print("Création de la partition terminée")

add_part(g,tab)




# 3/ Correction du graphe
  
    # calculer le dregre max
def fusion ( i : int , j : int ):
    # Fusionne le noeud i et j, en attibuant remplaçant j par i dans les voisins de celui-ci
    for k,v in g[j].items():
        if (k!=i and not(k in g[i])):
            g[i][k]=v
        g[k][i]=v
        del g[k][j]
        
    noeud_valide[j]=False
    
def apptab(x : int, y : int ):
    # part[x][y] isn't out of range
    return(x>=0 and y>=0 and x<nlong and y<nlat)


def classement(g:dict, t : list , eps : float):
    ## Classe le dictionnaire g en recquérant les fusions proches à eps près
    ## Précondition : eps<eta
    for k in g.keys():
        if noeud_valide[k]: #Teste si k n'a pas déjà été fusionné 
            X=k
            a_fusionner=[]
            # a_fusionner U {X} est l'ensemble des noeuds à distance < esp déjà trouvés 
            # X est celui d'identifiant minimal
            n,e=tabclasse[k]

            for i in range(-1,2):
                for j in range(-1,2):
                    if apptab(n+i,e+j):
                        for z in partnoeuds[n+i][e+j]:
                            if noeud_valide[z] and distance(t[z],t[k])<eps and z!=X:
                                if z<X:
                                    a_fusionner.append(X)
                                    X=z
                                else: 
                                    a_fusionner.append(z)
            for i in a_fusionner:
                fusion(X,i)
                assert(not(noeud_valide[i]))

classement(g,tab,0.9*eta) # Attention eps < eta
#affiche_graphe(g,tab)
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
            o=int(table[k])
            gf[o]={}
            for kk,vv in v.items():
                if table[kk]!=-1:
                    p=int(table[kk])
                    gf[o][p]=vv


renum(noeud_valide,gf_coord,g,g_final)

print("Taille avant fusion:",len(g),"Taille après fusion",len(g_final))

affiche_graphe(g_final,gf_coord)

