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

	### We make sure the wavefunction is normalized 
	wf = wf/np.sqrt(norm)

	for i in range(4):
	    out[i,...] = np.real( np.sum( np.conjugate(wf) * np.tensordot(spin_one_matrices[i],wf,axes=(1,0)), axis=0) )/norm

	return out

### This evaluates the magnetization <S> on each site 
### Returns a tensor <S>[c,x,y] with c = 0,1,2,3 the component 
def charge_squared(wf):
	norm = overlap(wf,wf)
	### We make sure the wavefunction is normalized 
	wf = wf/np.sqrt(norm)
	out = np.real( np.sum( np.conjugate(wf) * np.tensordot( (spin_one_matrices[3])@(spin_one_matrices[3]), wf,axes=(1,0)),axis=0)/norm)

	return out  

### Evaluates the total energy of an ansatz wavefunction for given Ec and Ej parameters 
### These may be arrays 
def energy(wf,Ec,Ej):

	### We make sure the wavefunction is normalized 
	norm = overlap(wf,wf)
	wf = wf/np.sqrt(np.real(norm))

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
	renorm = overlap(wf,wf)

	wf = wf/np.sqrt(np.real(renorm))

	return wf, res.fun

### This method starts from superfluid state and turns on the charge to anneal to the final state 
def find_GS_anneal_Ec(Lx,Ly,Ec,Ej):
	### We define a function which computes the energy given a flattened wavefunction


	### First we come up with an annealing schedule 
	neps = 25
	epsilons = np.linspace(0.,1.,neps)

	wf0 = initialize_SF(Lx,Ly,0.)
	wf0 = wf0.flatten()

	for i in range(neps):
		Ec_eps = epsilons[i]*Ec


		### This takes a flattened wavefunction and computes the energy of this state by unflattening and evaluating energy 
		def e_func(wf_flat):
			### First flatten the wavefunction
			wf = wf_flat.reshape((3,Lx,Ly))

			e = energy(wf,Ec_eps,Ej)

			return e 


		res = opt.basinhopping(e_func,wf0)

		wf0 = res.x
		renorm = overlap(wf0,wf0)
		wf0 = wf0/np.sqrt(np.real(renorm))


	wf = wf0
	renorm = overlap(wf,wf)

	wf = wf/np.sqrt(np.real(renorm))

	return wf


#################################
####### GS Time evolution #######
#################################

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
	if 
	wf0,e = find_GS(Lx,Ly,Ec,Ej)

	X0 = wf0.flatten()

	### Now we solve dynamics starting from this, and hopefully find no dynamics in absence of external perturbation
	sol = intg.solve_ivp(eom,(times[0],times[-1]),X0,t_eval=times,args=(Lx,Ly,Ec,Ej),max_step=0.01)

	### Now we reshape the output 
	wf_vs_t = sol.y.reshape((3,Lx,Ly,len(sol.t)))

	### Return wf and times 
	return wf_vs_t

#####################################
####### Quench time evolution #######
#####################################

### This is the equation of motion function in presence of a quench of local vorticity field on plaquette (00) 
### Accepts the flattened wavefunction as an argument
def eom_quench(t,X,Lx,Ly,Ec,Ej,quench_function):
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

	### Now we incorporate the quench
	### This couples in the following way
	### Depending on the quench function we apply a hopping phase to the plaquette (0,0) -> (1,0) -> (1,1) -> (0,1)
	sites = [ np.array([0,0]), np.array([1,0]) , np.array([1,1]), np.array([0,1]) ] ### Sites in the plaquette we traverse in a loop

	### The quench function controls the flux inserted, and the Hamiltonian coefficient remains EJ
	### So the josephson terms for the sites on the plaquette will be zeroed out and re-made specifically to incorporate the flux inserted 

	vorticity_eom = np.zeros_like(josephson_eom)

	for i in range(len(sites)):
		r = sites[i]
		x = r[0]
		y = r[1]
		fwd = sites[(i+1)%len(sites)] ### The forward neighbor on the loop
		bwd = sites[i-1] ### The previous neighbor on the loop

		### These should be the other two neighbors of the site on the plaquette
		### We reflect the neighbor which is in the plaquette through the site r which then should lie out of the plaquette 
		out_1 = (2*r - fwd)%np.array([Lx,Ly])
		out_2 = (2*r - bwd)%np.array([Lx,Ly])

		### We generate the correct Curie Weiss field including the flux inserted 
		flux_phase = np.exp(1.j*quench_function(t)) ### This describes the size and time-dependence of the flux inserted as a function of time 
		complex_curie_weiss = -0.5*Ej*flux_phase*(m[1,fwd[0],fwd[1]]+1.j*m[2,fwd[0],fwd[1]]) 
		complex_curie_weiss += -0.5*Ej*np.conjugate(flux_phase)*(m[1,bwd[0],bwd[1]]+1.j*m[2,bwd[0],bwd[1]]) 
		complex_curie_weiss += -0.5*Ej*(m[1,out_1[0],out_1[1]] + 1.j*m[2,out_1[0],out_1[1]]) 
		complex_curie_weiss += -0.5*Ej*(m[1,out_2[0],out_2[1]] + 1.j*m[2,out_2[0],out_2[1]]) 

		### Now we form this in to the appropriate matrix 
		eom_matrix = np.real(complex_curie_weiss)*spin_one_matrices[1] + np.imag(complex_curie_weiss)*spin_one_matrices[2]

		vorticity_eom[:,x,y] += -1j*np.tensordot(eom_matrix, wf[:,x,y],axes=[1,0])
		josephson_eom[:,x,y] *= 0.

	### We now add the correct eom contributions, which are only nonzero on the plaquette sits, to the josephson terms which are zeroed on the plaquette sites but otherwise correct
	josephson_eom += vorticity_eom

	dXdt += josephson_eom.flatten()

	return dXdt 

### This implements a simple Ramsey type flux quench which turns on to value flux at time tR  
def ramsey_flux_quench(t,flux,tR):
	if t >tR:
		return flux

	else:
		return 0.

### This computes the wavefunction starting from the ground state and then in response to a quench of the local vorticity
def solve_eom_quench(Lx,Ly,Ec,Ej,times,quench_function,wf0=None):
	### First we find the ground state 
	### We will first find the GS unless we are explicitly passed one as a seed state (useful for symmetry broken GS which may have have orthogonal degenerate GS)
	if wf0 is None:
		wf0,e = find_GS(Lx,Ly,Ec,Ej)

	X0 = wf0.flatten()

	### Now we solve dynamics starting from this, and hopefully find no dynamics in absence of external perturbation
	sol = intg.solve_ivp(eom_quench,(times[0],times[-1]),X0,t_eval=times,args=(Lx,Ly,Ec,Ej,quench_function),max_step=0.01)

	### Now we reshape the output 
	wf_vs_t = sol.y.reshape((3,Lx,Ly,len(sol.t)))

	### Return wf and times 
	return wf_vs_t



####################
####### Main #######
####################



















