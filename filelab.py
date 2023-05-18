
# Python Program illustrating
# pyplot.colorbar() method
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
  
popL=[4,7,17,30,24,2]


N = 30
  
# colormap
cmap = plt.get_cmap('jet', N)
  
fig, ax1 = plt.subplots()
  

    ax1.plot(x, y, c=cmap(popL[i]))
  
plt.xlabel('x-axis')
plt.ylabel('y-axis')
  
# Normalizer
norm = mpl.colors.Normalize(vmin=0, vmax=N)
  
# creating ScalarMappable
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
  
plt.colorbar(sm, ticks=np.linspace(0, 2, N))
  
  
plt.show()


def affiche_graphe( d : dict, tab : list,immeuble : list):
    #fig, ax = plt.subplots()
    
    N = max
  
    # colormap
    cmap = plt.get_cmap('jet', N)
    c=0
    for k,v in d.items():
        for j in v.keys():
            
            x=[tab[j][0],tab[k][0]]
            y=[tab[j][1],tab[k][1]]
            ax1.plot(x, y, c=cmap(popL[i]))
            c+=1
            
    # Normalizer
    norm = mpl.colors.Normalize(vmin=0, vmax=N)
    
    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    plt.colorbar(sm, ticks=np.linspace(0, 2, N))
    plt.show()                  
