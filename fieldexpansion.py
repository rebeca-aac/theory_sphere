# Este programa obtiene la expansión en armónicos esféricos del campo eléctrico usando
# la librería PyMieScatt para obtener los coeficientes an, bn, cn y dn. Con ello
# se desarrolla la expresión completa en términos de funciones especiales

import numpy as np
import matplotlib.pyplot as plt
from PyMieScatt import MieQ
from PyMieScatt.Mie import Mie_ab, Mie_cd
from scipy.special import spherical_jn, spherical_yn, lpmv

c=3E8                                              # velocidad de la luz
wp=5.89E15                                         # frecuencia del plasma en rad/s
gam=wp/100                                         # factor de amortiguamiento

#------------------------------------------------- Función dielécrica de Drude
def epsilon_drude(w,gam):
    return 1 - (wp**2) / (w * (w + 1j * gam))
#------------------------------------------------- 

#------------------------------------------------- Función para cambio de coordenadas
def spherical_to_cartesian(Er, Etheta, Ephi, theta, phi):
    Ex = (Er * np.sin(theta) * np.cos(phi)
         + Etheta * np.cos(theta) * np.cos(phi)
         - Ephi * np.sin(phi))
    Ey = (Er * np.sin(theta) * np.sin(phi)
         + Etheta * np.cos(theta) * np.sin(phi)
         + Ephi * np.cos(phi))
    Ez = (Er * np.cos(theta) - Etheta * np.sin(theta))
    return Ex, Ey, Ez
#-------------------------------------------------

#------------------------------------------------- Funciones auxiliares pi_n y tau_n
def pitau_n(n, coord_theta):
    # Nota: se agregan signos menos en pi y tau debido 
    # a una diferencia de convención con SciPy de lpmv
    
    # VERSIÓN ORIGINAL
    #P_n   = lpmv(1, n, np.cos(coord_theta))
    #P_n_1 = lpmv(1, n-1, np.cos(coord_theta))
    #tau_n = -(n*np.cos(coord_theta)*P_n - (n+1)*P_n_1)/np.sin(coord_theta)
    #if n==0:
    #    pi_n=0
    #if n==1:
    #    pi_n=1
    #else:
    #    pi_n  = -P_n / np.sin(coord_theta)

    # VERSIÓN CHAT
    if np.isclose(coord_theta, 0):
        pi_n = n*(n+1)/2

    elif np.isclose(coord_theta, np.pi):
        pi_n = (-1)**(n+1)*n*(n+1)/2

    else:
        P_n = lpmv(1, n, np.cos(coord_theta))
        pi_n = -P_n / np.sin(coord_theta)

    P_n = lpmv(1, n, np.cos(coord_theta))
    P_n_1 = lpmv(1, n-1, np.cos(coord_theta))

    tau_n = -(n*np.cos(coord_theta)*P_n
              - (n+1)*P_n_1) / np.sin(coord_theta)
    
    return pi_n, tau_n
#-------------------------------------------------

#------------------------------------------------- Componentes del campo eléctrico incidente
def mie_incident_insidesphere_efield(coord_r,coord_theta,coord_phi,n_particle,wavelength,nmax):
    # Nota: el n-ésimo término de la expansión para el campo interno es
        # E1_n = En [Mo1n−iNe1n]
        # Mo_1n = cos(ϕ)*πn(cosθ)*jn(kr)*vu(θ)  −  sen(ϕ)*τn(cosθ)*jn(kr)*vu(ϕ)
        # Ne_1n = cos(ϕ)*n(n +1)*sen(θ)*πn(cosθ)*[jn(kr)/kr]*vu(r)
        #         + cos(ϕ)*τn(cosθ)*([(kr)jn(kr)]′/kr)*vu(θ) − sen(ϕ)*πn(cosθ)*([(kr)jn(kr)]′/kr)*vu(ϕ)

    k1r = 2*np.pi*n_particle*coord_r/wavelength     # k1=2πn/λ
    Er, Etheta, Ephi = 0j, 0j, 0j

    for n in range(1, nmax):
        jn      = spherical_jn(n, k1r)
        djn     = n/k1r * jn - spherical_jn(n+1, k1r)
        krjnkr = jn + k1r*djn
        pi_n, tau_n = pitau_n(n, coord_theta)
        En = (1j)**n * (2*n + 1) / (n*(n+1))

        # -En*i*cos(ϕ)*n(n +1)*sen(θ)*πn(cosθ)*[jn(kr)/kr]
        Er += -En*1j*np.cos(coord_phi)*n*(n+1)*np.sin(coord_theta)*pi_n*jn/k1r
        # En*cos(ϕ)*πn(cosθ)*jn(kr)-En*i*cos(ϕ)*τn(cosθ)*([(kr)jn(kr)]′/kr)    
        Etheta += En * np.cos(coord_phi)*(pi_n*jn -1j*tau_n*krjnkr/k1r)
        # -En*sen(ϕ)*τn(cosθ)*jn(kr)+ En*i*sen(ϕ)*πn(cosθ)*([(kr)jn(kr)]′/kr)
        Ephi   += En * np.sin(coord_phi)*(1j*pi_n*krjnkr/k1r-tau_n*jn)
    return Er, Etheta, Ephi

def mie_incident_outsidesphere_efield(coord_r,coord_theta,coord_phi,n_medium,wavelength,nmax):
    # Nota: el n-ésimo término de la expansión para el campo interno es
        # E1_n = En [Mo1n−iNe1n]
        # Mo_1n = cos(ϕ)*πn(cosθ)*jn(kr)*vu(θ)  −  sen(ϕ)*τn(cosθ)*jn(kr)*vu(ϕ)
        # Ne_1n = cos(ϕ)*n(n +1)*sen(θ)*πn(cosθ)*[jn(kr)/kr]*vu(r)
        #         + cos(ϕ)*τn(cosθ)*([(kr)jn(kr)]′/kr)*vu(θ) − sen(ϕ)*πn(cosθ)*([(kr)jn(kr)]′/kr)*vu(ϕ)

    kr = 2*np.pi*n_medium*coord_r/wavelength     # k=2πn/λ
    Er, Etheta, Ephi = 0j, 0j, 0j

    for n in range(1, 100):
        jn      = spherical_jn(n, kr)
        djn     = n/kr * jn - spherical_jn(n+1, kr)
        krjnkr = jn + kr*djn
        pi_n, tau_n = pitau_n(n, coord_theta)
        En = (1j)**n * (2*n + 1) / (n*(n+1))

        # -En*i*cos(ϕ)*n(n +1)*sen(θ)*πn(cosθ)*[jn(kr)/kr]
        Er += -En*1j*np.cos(coord_phi)*n*(n+1)*np.sin(coord_theta)*pi_n*jn/kr
        # En*cos(ϕ)*πn(cosθ)*jn(kr)-En*i*cos(ϕ)*τn(cosθ)*([(kr)jn(kr)]′/kr)    
        Etheta += En * np.cos(coord_phi)*(pi_n*jn -1j*tau_n*krjnkr/kr)
        # -En*sen(ϕ)*τn(cosθ)*jn(kr)+ En*i*sen(ϕ)*πn(cosθ)*([(kr)jn(kr)]′/kr)
        Ephi   += En * np.sin(coord_phi)*(1j*pi_n*krjnkr/kr-tau_n*jn)
    return Er, Etheta, Ephi

#------------------------------------------------- Componentes del campo eléctrico interno
def mie_internal_efield(cn,dn,coord_r,coord_theta,coord_phi,n_particle,wavelength,nmax):
    # Nota: el n-ésimo término de la expansión para el campo interno es
        # E1_n = En [ cn*Mo_1n - idn*Ne_1n ]
        # Mo_1n = cos(ϕ)*πn(cosθ)*jn(kr)*vu(θ)  −  sen(ϕ)*τn(cosθ)*jn(kr)*vu(ϕ)
        # Ne_1n = cos(ϕ)*n(n +1)*sen(θ)*πn(cosθ)*[jn(kr)/kr]*vu(r)
        #         + cos(ϕ)*τn(cosθ)*([(kr)jn(kr)]′/kr)*vu(θ) − sen(ϕ)*πn(cosθ)*([(kr)jn(kr)]′/kr)*vu(ϕ)

    k1r = 2*np.pi*n_particle*coord_r/wavelength     # k1=2πn/λ
    Er, Etheta, Ephi = 0j, 0j, 0j

    for n in range(1, nmax):
        jn      = spherical_jn(n, k1r)
        djn     = n/k1r * jn - spherical_jn(n+1, k1r)
        krjnkr = jn + k1r*djn
        pi_n, tau_n = pitau_n(n, coord_theta)
        En = (1j)**n * (2*n + 1) / (n*(n+1))

        # -En*idn*cos(ϕ)*n(n +1)*sen(θ)*πn(cosθ)*[jn(kr)/kr]
        Er += -En*1j*dn[n-1]*np.cos(coord_phi)*n*(n+1)*np.sin(coord_theta)*pi_n*jn/k1r
        # En*cn*cos(ϕ)*πn(cosθ)*jn(kr)-En*idn*cos(ϕ)*τn(cosθ)*([(kr)jn(kr)]′/kr)    
        Etheta += En * np.cos(coord_phi)*(cn[n-1]*pi_n*jn -1j * dn[n-1]*tau_n*krjnkr/k1r)
        # -En*cn*sen(ϕ)*τn(cosθ)*jn(kr)+ En*idn*sen(ϕ)*πn(cosθ)*([(kr)jn(kr)]′/kr)
        Ephi   += En * np.sin(coord_phi)*(1j*dn[n-1]*pi_n*krjnkr/k1r-cn[n-1]*tau_n*jn)
    return Er, Etheta, Ephi

#------------------------------------------------- Componentes del campo eléctrico dispersado
def mie_scattered_efield(an,bn,coord_r,coord_theta,coord_phi,n_medium,wavelength,nmax):
    # Nota: el n-ésimo término de la expansión para el campo dispersado es
        # E1_n = En [ ian*Ne_1n − bn*Mo_1n ]
        # Mo_1n = cos(ϕ)*πn(cosθ)*hn(kr)*vu(θ)  −  sen(ϕ)*τn(cosθ)*hn(kr)*vu(ϕ)
        # Ne_1n = cos(ϕ)*n(n +1)*sen(θ)*πn(cosθ)*[hn(kr)/kr]*vu(r)
        #         + cos(ϕ)*τn(cosθ)*([(kr)hn(kr)]′/kr)*vu(θ) − sen(ϕ)*πn(cosθ)*([(kr)hn(kr)]′/kr)*vu(ϕ)

    kr = 2*np.pi*n_medium*coord_r/wavelength     # k=2πn/λ
    Er, Etheta, Ephi = 0j, 0j, 0j

    for n in range(1, nmax):
        jn = spherical_jn(n, kr)
        yn = spherical_yn(n, kr)
        hn = jn + 1j*yn
        djn= n/kr * jn - spherical_jn(n+1, kr)
        dyn= n/kr * yn - spherical_yn(n+1, kr)
        dhn= djn + 1j*dyn
        krhnkr = hn + kr*dhn
        pi_n, tau_n = pitau_n(n, coord_theta)
        En = (1j)**n * (2*n + 1) / (n*(n+1))

        # En*ian*cos(ϕ)*n(n +1)*sen(θ)*πn(cosθ)*[hn(kr)/kr]
        Er += En*1j*an[n-1]*np.cos(coord_phi)*n*(n+1)*np.sin(coord_theta)*pi_n*hn/kr
        # En*ian*cos(ϕ)*τn(cosθ)*([(kr)hn(kr)]′/kr) - En*bn*cos(ϕ)*πn(cosθ)*hn(kr)
        Etheta += En * np.cos(coord_phi)*(1j*an[n-1]*tau_n*krhnkr/kr - bn[n-1]*pi_n*hn)
        # - En*ian*sen(ϕ)*πn(cosθ)*([(kr)hn(kr)]′/kr) + En*bn*sen(ϕ)*τn(cosθ)*hn(kr)
        Ephi   += En * np.sin(coord_phi)*(bn[n-1]*tau_n*hn - 1j*an[n-1]*pi_n*krhnkr/kr)
    return Er, Etheta, Ephi

# Radio y longitud de onda en nm
a = 50
wavelength = 520
w0 = 2*np.pi*c/(wavelength*10**(-9))

# Índice de refracción complejo de la esfera y del medio externo
n=np.real(np.sqrt(epsilon_drude(w0,gam)))
k=np.imag(np.sqrt(epsilon_drude(w0,gam)))
n_particle = n + 1j*k
n_medium = 1

m = n_particle/n_medium

# Parámetro de tamaño
d = 2*a
x = np.pi * d * n_medium / wavelength
nmax = int(2+x+4*x**(1/3))
print(nmax)

# Coeficientes del campo externo
an, bn = Mie_ab(m, x)
# Coeficientes del campo interno
cn, dn = Mie_cd(m, x)

# ------------------------------------ Hasta aquí, los coeficientes son calculados correctamente (parecen converger 11/nov/26)

Nx = Ny = Nz = 400   # resolución del mapa
x_vals = np.linspace(-70, 70, Nx)
y_vals = np.linspace(-70, 70, Ny)
z_vals = np.linspace(-70, 70, Nz)

Ex_map = np.zeros((Nz, Ny))
#Ex_map = np.zeros((Nx, Ny))

for iy, y in enumerate(y_vals):
    for iz, z in enumerate(z_vals):
    #for ix, x in enumerate(x_vals):
        # Plano YZ (con x=0)
        r = np.sqrt(y**2 + z**2)
        theta = np.arccos(z/r)
        if y > 0:
            phi = np.pi/2
        else:
            phi = 3*np.pi/2

        # Plano XY (con z=0)
        #r = np.sqrt(x**2 + y**2)
        #theta = np.pi/2
        #phi = np.arctan2(y,x)

        if r < a:
            #Er, Eth, Eph = mie_internal_efield(cn,dn,r,theta,phi,n_particle,wavelength,nmax)
            Er, Eth, Eph = mie_incident_outsidesphere_efield(r,theta,phi,n_medium,wavelength,nmax)
            Ex, Ey, Ez = spherical_to_cartesian(Er, Eth, Eph, theta, phi)
            Ex_map[iz, iy] = np.abs(Ex)
        elif r > a:
            #Er, Eth, Eph = mie_scattered_efield(an,bn,r,theta,phi,n_medium,wavelength,nmax)
            Er, Eth, Eph = mie_incident_outsidesphere_efield(r,theta,phi,n_medium,wavelength,nmax)
            Ex, Ey, Ez = spherical_to_cartesian(Er, Eth, Eph, theta, phi)
            Ex_map[iz, iy] = np.abs(Ex)

plt.figure(figsize=(8,6))
plt.imshow(Ex_map, extent=[y_vals[0], y_vals[-1], z_vals[0], z_vals[-1]],
           origin='lower', cmap='inferno')
plt.colorbar(label='|Ex|')
plt.xlabel("y (nm)")
plt.ylabel("z (nm)")
plt.title("Mapa de |Ex| en el plano x = 0")
plt.tight_layout()
plt.show()
