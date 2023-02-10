import math as m 

lat=m.radians(43.54946816425778)
lat_t=m.radians(43.54946816425778)
long=m.radians(1.37651356077766)
long_t=m.radians(1.506607510636402)
x=m.cos(lat)*m.cos(lat_t)*((m.sin((long-long_t)/2))**2)
r=6371*(10**3)
y=m.asin(m.sqrt(x))
d=2*r*y

print(y)
print("res:",d)


p=2*r*m.asin(m.sqrt((m.sin((lat-lat_t)/2)**2)+m.cos(lat)*m.cos(lat_t)*(m.sin((long-long_t)/2))**2))
print(p)
a=[long_t,lat_t]
b=[long,lat]

def distance (a,b):
    #Calcule d(a,b) en m
    lat=m.radians(a[1])
    long=m.radians(b[0])
    r=6371*(10**3)
    lat_t=m.radians(b[1])
    long_t=m.radians(b[0])
    p=2*r*m.asin(m.sqrt((m.sin((lat-lat_t)/2)**2)+m.cos(lat)*m.cos(lat_t)*(m.sin((long-long_t)/2))**2))
    return p

f=distance(a,b)
print("rresdisatn:",f)


