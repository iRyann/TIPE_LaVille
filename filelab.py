
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


#4/ Traitement du graphe 

## --------- Ajout de la population ------------ #

immeubleL = [] ## Liste des immeubles  
poidsL = [] ##
 

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
def ajoute_population( g : dict , poids_immeuble : list , immeuble : list) :
## g est une copie du graphe
    for k,v in g.items() :
        for kk,vv in v.items():
            if vv>=0:
                t=proba_rue()
                pt=0 ##
                px=0 ##
                poids=0
                temp=0
                while(px<vv):
                    
                    match t:
                        case 0:
                            px=10/vv
                            pt=random.uniform(0,0.5*px)
                            px+=pt
                            poids=random.randint(10,20)
                        case 1:
                            px=10/vv
                            pt=random.uniform(0,0.5*px)
                            px+=pt
                            poids=random.randint(20,30)
                        case 2:
                            px=5/vv
                            pt=random.uniform(0,0.8*px)
                            px+=pt
                            poids=random.randint(10,20)
                    if temp+px<vv:
                        immeuble.append((k,kk,temp+px))
                        poids_immeuble.append(poids)
                        temp=temp+px
                        
                    else: 
                        break 


def genere_graphe (rayon_g : int , nom : str ) : 
# 
  
    #On récupère les données dans f grâce à la requête requete2
    t=recupere_data(f,con,requete2,rayon_g)
    
    #On complete le graphe g grâce aux données récupérées dans t
    #en retenant les coordonnées de chaque noeud de g dans tab
    ajoute_noeud(g,t,tab)
    print("ici")
    partition_vide(partnoeuds,nlat,nlong) # On crée la partition vide
    add_part(g,tab)
    print("ici")
    classement(g,tab,0.9*eta) # Attention eps < eta
    renum(noeud_valide,gf_coord,g,g_final)
    print("ici")
    ajoute_population(g_final,poidsL,immeubleL)
    print("icifin")
    fichier = open(nom+".json","wt")
    fichier.write(json.dumps(g_final))
    fichier.close()
    fichier1 = open(nom+"coord.json","wt")
    fichier1.write(json.dumps(gf_coord))
    fichier1.close()


    print("Le graphe",nom,"a bien été crée.")

def recup_graphe(nom : str):
    #Retourne (graphe,liste des coordonnées des points du graphe)
    try :
        fichier = open(nom+".json","rt")
        graphe = json.loads(fichier.read(), object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
        fichier.close()

        fichier = open(nom+"coord.json","rt")
        g_coord = json.loads(fichier.read())
        fichier.close()

        return (graphe,g_coord)
    except FileNotFoundError as erreur :
        print("Le fichier intitulé",nom,"n'existe pas :",erreur)
    

genere_graphe(50,"graphe")


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
                
                plt.scatter('x', 'y', c='color', s='taille', data=data)
            # -------------------------------------
            c+=1
    
    cbar= plt.colorbar()

    cbar.set_label("Densité de population", labelpad=+1)




from matplotlib import pyplot as plt

# Scatter plot (plot à point avec palette de couleur par défaut Viridis()

# scatter(x,y,c=color, s=taille)






plt.show()





def affiche_graphe( d : dict, tab : list,immeuble : list,poids_im : list, poids : list):
    fig, ax = plt.subplots()
    c=0 # Compteur de parcours des listes
    c_im=0
    for k,v in d.items():
        for j in v.keys():
            
            # On trace la rue -------------------
            x=[tab[j][0],tab[k][0]]
            y=[tab[j][1],tab[k][1]]
            e=tab[j][0]-tab[k][0]
            if e==0:
                e=10**(-16)

            ang=(tab[j][1]-tab[k][1])/(e)
            xx = np.linspace(tab[j][0],tab[k][0], 100)
            yy = ang*(xx-tab[k][0])+tab[k][1]
            
            
            ax.scatter(xx, yy, s=0.02, c='black')
            

       
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
            
            plt.plot(xx, yy, linewidth=0.5, c=popL[c])
                       
            c+=1
    
    cbar= plt.colorbar(popL)

    cbar.set_label("Densité de population", labelpad=+1)
       
    plt.show()