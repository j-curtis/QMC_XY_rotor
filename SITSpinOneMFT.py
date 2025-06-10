### Mean field variational dynamics for spin-1 XY model in 2D 
### Jonathan Curtis 
### 05/26/25

import numpy as np

from scipy import integrate as intg
from scipy import optimize as opt

import time
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as mclr

rng = np.random.default_rng()

spin_one_matrices = [ np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=complex), 1./np.sqrt(2.)*np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]],dtype=complex),1./np.sqrt(2.)*np.array([[0.,-1.j,0.],[1.j,0.,-1.j],[0.,1.j,0.]],dtype=complex),np.array([[1.,0.,0.],[0.,0.,0.],[0.,0.,-1.]],dtype=complex) ]


##########################################
####### Wavefunction initilizaiton #######
##########################################

### This initializes a wavefunction on Lx x Ly grid in the Mott state 
def initialize_Mott(Lx,Ly):
    wf = np.zeros((3,Lx,Ly),dtype=complex) ### 3 components for each site, LxxLy sites 
    
    ### According to the parameterization using the spin-one matrices above the charge operator is S^z and thus the [1] component is  the |0> Fock state 
    wf[1,...] = 1.+0.j 
    
    return wf 

### This initializes a wavefunction on Lx x Ly grid in the superfluid state with a chosen phase 
def initialize_SF(Lx,Ly,phase):
    wf = np.zeros((3,Lx,Ly),dtype=complex) ### 3 components for each site, LxL sites
    
    wf[0,...] = 0.5*np.exp(1.j*phase)
    wf[1,...] = 0.5*np.sqrt(2.)
    wf[2,...] = 0.5*np.exp(-1.j*phase)
    
    return wf

###################################
####### Compute observables #######
###################################
### This takes the wavefunction overlap and returns it resolved in space
def overlap(w1,w2):
    return np.sum( np.conjugate(w1)*w2,axis=0)

### This evaluates the magnetization <S> on each site 
### Returns a tensor <S>[c,x,y] with c = 0,1,2,3 the component 
def magnetization(wf):
    out = np.zeros((4,*wf.shape[1:]))
    
    norm = np.real(overlap(wf,wf))
    
    for i in range(4):
        out[i,...] = np.real( np.sum( np.conjugate(wf) * np.tensordot(spin_one_matrices[i],wf,axes=(1,0)), axis=0) )/norm
    
    return out

### This evaluates the magnetization <S> on each site 
### Returns a tensor <S>[c,x,y] with c = 0,1,2,3 the component 
def charge_squared(wf):
    norm = overlap(wf,wf)
    out = np.real( np.sum( np.conjugate(wf) * np.tensordot( (spin_one_matrices[3])@(spin_one_matrices[3]), wf,axes=(1,0)),axis=0)/norm)
    
    return out  

### Evaluates the total energy of an ansatz wavefunction for given Ec and Ej parameters 
### These may be arrays 
def energy(wf,Ec,Ej):
	charging_energy = np.sum( Ec*charge_squared(wf) )
	m = magnetization(wf)
	Josephson_energy = -0.5*Ej*sum([ 
		np.sum( m[1,...]*np.roll(m[1,...], shift=s, axis=[0,1] ) ) + np.sum( m[2,...]*np.roll(m[2,...], shift=s, axis=[0,1] ) )
		for s in [ [0,1], [1,0] ] ])

	return np.real( charging_energy + Josephson_energy )

### Evalutes the mean superfluid density 
def SF_OP(wf):
	m = magnetization(wf) ### this is the magnetization at each point in space 

	op = np.mean( m[1,...] + 1.j*m[2,...])

	return op

####################################
####### Finding ground state #######
####################################

### Basin hopping to find the ground state
### Optional to pass initial guess, which is useful for annealing
def find_GS(Lx,Ly,Ec,Ej,wf0=None):
    ### We define a function which computes the energy given a flattened wavefunction

    ### This takes a flattened wavefunction and computes the energy of this state by unflattening and evaluating energy 
    def e_func(wf_flat):
    	### First flatten the wavefunction
    	wf = wf_flat.reshape((3,Lx,Ly))

    	e = energy(wf,Ec,Ej)

    	return e 

    if wf0 is None:
    	wf0 = initialize_Mott(Lx,Ly)

    wf0 = wf0.flatten()

    res = opt.basinhopping(e_func,wf0)

    wf = res.x.reshape((3,Lx,Ly))

    return wf, res.fun


##############################
####### Time evolution #######
##############################

### This is the equation of motion function 
### Accepts the flattened wavefunction as an argument
def eom(t,X,Lx,Ly,Ec,Ej):
    ### First we unflatten X 
    wf = X.reshape((3,Lx,Ly))
    
    ### We now compute the equation of motion
    ### First term is the local charging energy
    ### This is -i Ec[x,y] Sz^2 psi[x,y]
    
    charging_eom = -1.j*np.tensordot( (spin_one_matrices[3])@(spin_one_matrices[3]), wf,axes=(1,0))*Ec
    
    dXdt = charging_eom.flatten()

    ### Now we have the Josephson contributions
    ### These are obtained as S.MF
    ### Where MF = -0.5*Ej*( m[x+1,y] + m[x-1,y]+m[x,y+1]+m[x,y-1]) 

    m = magnetization(wf)

    curie_weiss = -0.5*Ej*( np.roll(m,shift=[0,1,0],axis=[0,1,2]) + np.roll(m,shift=[0,-1,0],axis=[0,1,2])+np.roll(m,shift=[0,0,1],axis=[0,1,2]) + np.roll(m,shift=[0,0,-1],axis=[0,1,2]))
    
    josephson_eom = -1.j*( np.tensordot(spin_one_matrices[1],wf,axes=[1,0]) * curie_weiss[1,...] + np.tensordot(spin_one_matrices[2] , wf,axes=[1,0])*curie_weiss[2,...] )

    dXdt += josephson_eom.flatten()
                 
    return dXdt 



def solve_eom_from_GS(Lx,Ly,Ec,Ej,times):
	### First we find the ground state 
	wf0,e = find_GS(Lx,Ly,Ec,Ej)

	X0 = wf0.flatten()

	### Now we solve dynamics starting from this, and hopefully find no dynamics in absence of external perturbation
	sol = intg.solve_ivp(eom,(times[0],times[-1]),X0,t_eval=times,args=(Lx,Ly,Ec,Ej),max_step=0.01)

	### Now we reshape the output 
	wf_vs_t = sol.y.reshape((3,Lx,Ly,len(sol.t)))

	### Return wf and times 
	return wf_vs_t



