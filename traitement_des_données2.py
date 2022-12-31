import sqlite3 as sql
import matplotlib.pyplot as plt
import math as m

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
    return (d<10000)

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
    return f


#On récupère les données dans f grâce à la requête requete2
t=recupere_data(f,con,requete2)




#---------------------------------------------------
#           Traitement des données
#---------------------------------------------------

g={} #Graphe
tab=[] #Coordonnées associées à un pt une graphe


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
            n+=1
        #Complétion du voisinnage
        if (len(t[i]))>1:
            for j in range (0,len(t[i])-1):
                # On ajoute les relations ( j <-> j+1 ) entre les sommets intermédiaire d'une même route
                ajoute_voisin(g,n-len(t[i])+j,(n-len(t[i])+j+1,distance(t[i][j],t[i][j+1])))

#On complete le graphe g grâce aux données récupérées dans t
#en retenant les coordonnées de chaque noeud de g dans tab
ajoute_noeud(g,t,tab)



# 2/ Correction du graphe


def remplace (i : int , j : int ) :
    #On ajoute les voisins de j à ceux de i
    for k,v in g[j].items():
        g[i][k]=v

    #On remplace j par i pour toutes les occurences de j
    for v in g.values():
        r={}
        dmoy=0
        c=1
        for k in v.keys():
            if k==j:
                r[k]=v[k]
                dmoy+=v[k]
                c+=1
        for k in r.keys():
            del v[k]
        v[i]=(dmoy/c)
        del c,dmoy,r






def classe_g (g : dict):
    c=0
    for i in range(0,len(g)):
        if bool(g[i]):
            for j in range(1+i,len(g)):
                if bool(g[j]) and distance(tab[i],tab[j])<1:
                    remplace(i,j)
                    g[j]={}
                    c+=1
                    print("Noeud",j,"remplacé par ",i,"Nombre total de fusion:",c)

#classe_g(g)

def affiche_graphe( d : dict, tab : list):
    fig, ax = plt.subplots()
    z=0
    for k,v in d.items():
        for j in v.keys():
            x=[tab[j][0],tab[k][0]]
            y=[tab[j][1],tab[k][1]]
            ax.step(x,y , linewidth=1)
            ax.scatter(x, y, s=1.5)
            z+=1
            print(k)
            if z==1000:
                break
        if z==1000:
                break
    plt.show()

affiche_graphe(g,tab)


