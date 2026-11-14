import numpy as np
import matplotlib.pyplot as plt
from PyMieScatt import MieQ
from PyMieScatt.Mie import Mie_ab, Mie_cd

c=3E8          # velocidad de la luz
wp=5.89E15     # frecuencia del plasma en rad/s
gam=wp/100     # factor de amortiguamiento

#------------------------------------------------- Función dielécrica de Drude
def epsilon_drude(w,gam):
    return 1 - (wp**2) / (w * (w + 1j * gam))
#------------------------------------------------- 

#------------------------------------------------- Calculadora de coeficientes Qsca y Qabs
def qscaqabs(a,n_particle,n_medium,m,wavelength,index):
    Qsca=[]
    Qabs=[]
    for i in index:
        x = 2*np.pi * a * n_particle / wavelength[i]
        # Coeficientes de eficiencia
        efficiency=MieQ(m[i], wavelength[i], 2*a, nMedium=n_medium, asDict=True, asCrossSection=False)
        Qsca.append(efficiency["Qsca"])
        Qabs.append(efficiency["Qabs"])

    return Qsca,Qabs
#------------------------------------------------- 

# Radio de la partícula en nm
radius=(10,20,50)

# Longitudes de onda en nm p/evaluar
wavelength=np.arange(250,805,5)
index=np.arange(0,111,1)
w0 = 2*np.pi*c/(wavelength*10**(-9))

# Índices de refracción para cada longitud de onda
n=np.real(np.sqrt(epsilon_drude(w0,gam)))
k=np.imag(np.sqrt(epsilon_drude(w0,gam)))
n_particle = n + 1j*k
n_medium = 1
m = n_particle/n_medium


for a in radius:
    eff_sca,eff_abs=qscaqabs(a,n_particle,n_medium,m,wavelength,index)
    plt.plot(wavelength, eff_abs, linestyle='-',label=f'{a} nm')

plt.xlabel("Longitud de onda (nm)")
plt.ylabel("Qabs")
plt.legend()
plt.tight_layout()
plt.show()