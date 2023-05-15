import sqlite3 as sql
import matplotlib.pyplot as plt
import math as m
import numpy as np
import random
import json

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
    return (d<100)

def distance (a,b):
    #Calcule d(a,b) en m
    lat=m.radians(a[1])
    long=m.radians(a[0])
    r=6371*(10**3)
    lat_t=m.radians(b[1])
    long_t=m.radians(b[0])
    d=2*r*m.asin(m.sqrt((m.sin((lat-lat_t)/2)**2)+m.cos(lat)*m.cos(lat_t)*(m.sin((long-long_t)/2))**2))
    return d

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

#---------------------------------------------------
#   Fonction d'affichage
#---------------------------------------------------

#def affiche_graphe( d : dict, tab : list,immeuble : list, poids_im : list):
    #fig, ax = plt.subplots()
    c=0
    c_im=0
    for k,v in d.items():
        for j in v.keys():

            x=[tab[j][0],tab[k][0]]
            y=[tab[j][1],tab[k][1]]
            e=tab[j][0]-tab[k][0]
            if e==0:
                e=10**(-16)


            xx = np.linspace(tab[j][0],tab[k][0], 100)
            yy = ((tab[j][1]-tab[k][1])/(e))*(xx-tab[k][0])+tab[k][1]
            ang = (tab[j][1]-tab[k][1])/(e)
            
            plt.plot(xx, yy, linewidth=0.5, c='green')
            
            # On ajoute les immeuble de la rue ---
            if c_im==c :
                
                xim_l=[]
                yim_l=[]
                area=[]
                
                for i in range (c_im,c_im+len(poids_im[c])):
                    (a,b,d)=immeuble[i]
                    xim_l.append(x[0]+d*np.sin(ang))
                    yim_l.append(x[1]+d*np.cos(ang)*(np.abs(ang)/ang))
                    area.append(poids_im[c][i-c_im])

                c_im = c + len(poids_im[c])

                data = {'x': np.array(xim_l),

                        'y': np.array(yim_l),

                        'color': np.array(area)}
                
                plt.scatter('x', 'y', c='color', data=data)
            # -------------------------------------
            
            c+=1
    
    cbar= plt.colorbar()

    cbar.set_label("Densité de population", labelpad=+1)
       
    plt.show()


def affiche_graphe( d : dict, tab : list,immeuble : list):
    #fig, ax = plt.subplots()
    c=0
    c_im=0
    for k,v in d.items():
        for j in v.keys():

            x=[tab[j][0],tab[k][0]]
            y=[tab[j][1],tab[k][1]]
            e=tab[j][0]-tab[k][0]
            if e==0:
                e=10**(-16)


            xx = np.linspace(tab[j][0],tab[k][0], 100)
            yy = ((tab[j][1]-tab[k][1])/(e))*(xx-tab[k][0])+tab[k][1]
            ang = (tab[j][1]-tab[k][1])/(e)
            
            plt.plot(xx, yy, linewidth=0.5)
                       
            c+=1
    
    
       
    plt.show()



#---------------------------------------------------#
#           Récupération des données                #
#---------------------------------------------------#

#-Base de donnée support ---------------------------
con = sql.connect('filaire_de_voirie.db')
requete2="""select geo_shape FROM "filaire-de-voirie" """


LGeopoints=[] # Tableau brut des données récupérées : géopoints


#---------------------------------------------------#
#           partnoeudsnement des données            #
#---------------------------------------------------#
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
t=recupere_data(LGeopoints,con,requete2)




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



def ajoute_noeud(g : dict, t : list, tab : list):
    n=0 #nombre de noeuds
    
    for i in range(0,len(t)):
        
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
print("Graphe final crée avec succès !")







#4/ Traitement du graphe 

## --------- Ajout de la population ------------ #

immeubleL = [] ## Liste des immeubles  
poidsL = [] # Poids de chaque immeuble
popL = [] # Nombre d'habitant par rue 
poidsLL = []
## Types de rues : 
# 0 : Administrative
# 1 : Résidentielle
# 2 : Commerçante

def proba_rue() :
# Retourne un type de rue     
    x=random.random()
    if x<0.5:
        y=random.random()
        if y<0.5:
            return 0
        else: return 1
    else: return 2



##dictinnaire d'arete traitée 
def ajoute_population( g : dict , poids_immeuble : list , immeuble : list, population : list) :
## g est une copie du graphe
    for k,v in g.items() :
        for kk,vv in v.items():
            print("Traitement de la rue :",k,"->",kk)
            if vv>=0:
                # On définit le type de la rue
                t=proba_rue()             
                poids_im=[]
                dist_parcours=0
                pop_rue=0
                while(dist_parcours<vv):
                    
                    poids=0
                    # Un immeuble tous les px +/- pt
                    pt=0 
                    px=0
                    match t:
                        case 0:
                            px=vv/5
                            pt=random.uniform(0,0.5*px)
                            px+=pt
                            poids=random.randint(10,20)
                        case 1:
                            px=vv/10
                            pt=random.uniform(0,0.5*px)
                            px+=pt
                            poids=random.randint(20,30)
                        case 2:
                            px=vv/7
                            pt=random.uniform(0,0.8*px)
                            px+=pt
                            poids=random.randint(10,20)
                    if dist_parcours+px<vv:
                        immeuble.append((k,kk,dist_parcours+px))
                        poids_im.append(poids)
                        poidsLL.append(poids)
                        pop_rue+=poids
                        dist_parcours=dist_parcours+px
                    
                    else: 
                        break 
                    poids_immeuble.append(poids_im)
                    population.append(pop_rue) 

ajoute_population(g_final,poidsL,immeubleL,popL)



affiche_graphe(g,tab,immeubleL)

# ---------------------------------------#
#               Résolution               #
#----------------------------------------#

import heapq


def succ (g,n) :
  u=n
  return g[u] 

def dijkstra (g,s,sf) :
  n = len(g)
  d = []
  for i in range (0,n) :
    d.append(1000)
  d[s]=0
  o=[]
  heapq.heappush(o, (d[s], s))
  while o != [] :
    (_,u) = heapq.heappop(o)
    for k in succ(g,u).keys() :
      v=int(k)
      if d[u]+succ(g,u)[k] < d[v] :
        temp = d[v]
        d[v] = d[u]+succ(g,u)[k]
        if temp == 1000 :
          heapq.heappush(o,(d[v],v))
  return d[sf]



def graph_with_station (g,s1,s2,d,ind_imm,l3):
  (s1_imm,s2_imm,d_imm) = l3[ind_imm]
  if (s1_imm == s1) and (s2_imm == s2) :
    if d_imm < d :
      ns1 = len(g)-1
      ns2 = s2_imm
      d = d-d_imm
    else :
      ns1 = s1_imm
      ns2 = len(g)-1
  elif (s1_imm == s2) and (s2_imm == s1) :
    if d_imm < (d_imm + (g[len(g)-1])[s2_imm] - d) :
      ns1 = s2_imm
      ns2 = len(g)-1
    else:
      ns1 = len(g)-1
      ns2 = s1_imm
      d = d - (g[len(g)-1])[s2_imm]
  else :
    ns1 = s1
    ns2 = s2
  new_g = {}
  for k,v in g.items() :
    new_g[k] = v.copy()
  sta=len(g)
  new_g[sta]={ns1:d,ns2:(g[ns1])[ns2]-d}
  (new_g[ns1])[sta]=d
  (new_g[ns2])[sta]=(g[ns1])[ns2]-d
  del (new_g[ns1])[ns2]
  del (new_g[ns2])[ns1]
  return new_g



def graph_with_imm (g,ind_imm,l3):
  new_g = {}
  for k,v in g.items() :
    new_g[k] = v.copy()
  ind_imm_gra=len(g)
  (s1,s2,d) = l3[ind_imm]
  ns1 = s1
  ns2 = s2
  new_g[ind_imm_gra]={ns1:d,ns2:(g[ns1])[ns2]-d}
  (new_g[ns1])[ind_imm_gra]=d
  (new_g[ns2])[ind_imm_gra]=(g[ns1])[ns2]-d
  del (new_g[ns1])[ns2]
  del (new_g[ns2])[ns1]
  return new_g



def min_dist (g,l,l3,p) :
  g_imm = graph_with_imm (g,p,l3)
  (s1,s2,d) = l[0]
  g0 = graph_with_station(g_imm,s1,s2,d,p,l3)
  res = dijkstra(g0,len(g0)-2,len(g0)-1)
  for i in range (1,len (l)) :
    (si1,si2,di) = l[i]
    gi = graph_with_station(g_imm,si1,si2,di,p,l3)
    res = min(res,dijkstra(gi,len(gi)-2,len(gi)-1))
  return res

def f (g,l,l2,l3) :
  res = 0
  som = 0
  for i in range (0,len(l3)) :
    res = res + (min_dist(g,l,l3,i)*l2[i])
  for i in range (0,len(l3)) :
    som = som + l2[i]
  res = res/som
  return res



def simulated_annealing(g,initial_state,l2,l3):
    """Peforms simulated annealing to find a solution"""
    initial_temp = 300
    final_temp = 1
    current_temp = initial_temp

    # Start by initializing the current state with the initial state
    current_state = initial_state
    solution = current_state

    while current_temp > final_temp:
        neighbor = get_neighbors(g,current_state,l2,current_temp)
        # print(neighbor)
        # Check if neighbor is best so far
        cost_diff = f(g,current_state,l2,l3)-f(g,neighbor,l2,l3)
        # print(cost_diff)
        # if the new solution is better, accept it
        if cost_diff > 0:
            solution = neighbor
        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
        else:
            if random.uniform(0, 1) > m.exp(cost_diff / current_temp) :
                solution = neighbor
        # decrement the temperature
        # print(solution)
        current_temp = current_temp * 0.999
        
        current_state = solution
    return solution




def get_neighbor_station(g,station,eps):
  """Returns neighbors of the argument state for your solution."""
  (s1,s2,d) = station
  ns1 = s1
  ns2 = s2
  carburant=random.uniform(0,eps)
  sens=random.randint(0,1)
  if sens == 1 :
    d = (g[ns1])[ns2] - d
    temp = ns1
    ns1 = ns2
    ns2 = temp
  while carburant != 0 :
    if carburant > (g[ns1])[ns2] - d :
      carburant = carburant - ((g[ns1])[ns2] - d)
      voisins = list(g[ns2].keys())
      temp=ns2
      ns2=random.choice(voisins)
      ns1=temp
      d = 0
    else :
      d = d + carburant
      carburant = 0
  return (ns1,ns2,d)


def get_neighbors(g,state,l2,eps):
  """Returns neighbors of the argument state for your solution."""
  res=[]
  for i in range (0,len(state)):
    res.append(get_neighbor_station(g,state[i],eps))
  
  return res

def best_answer (g,initial_state,l2,l3) :
  
  res = simulated_annealing(g,initial_state,l2,l3)
  
  print(res)
  for i in range (0,4) :
    new_res = simulated_annealing(g,initial_state,l2,l3)
    print(new_res)
    if f(g,new_res,l2,l3) < f(g,res,l2,l3) :
      res = new_res
  return res

tailleim=len(immeubleL)
si=[]
for i in range (0,3):
   o=random.randint(0,tailleim)
   si.append(immeubleL[o])


#answer = best_answer(g_final,si,poidsLL,immeubleL)²

#print(f(graphe,answer,a,imm))