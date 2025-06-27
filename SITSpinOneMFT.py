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

### This class will realize an instance of the XY model mean-field dynamics
class xymodel:
	spin_one_matrices = [ np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=complex), 1./np.sqrt(2.)*np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]],dtype=complex),1./np.sqrt(2.)*np.array([[0.,-1.j,0.],[1.j,0.,-1.j],[0.,1.j,0.]],dtype=complex),np.array([[1.,0.,0.],[0.,0.,0.],[0.,0.,-1.]],dtype=complex) ]

	### Class methods

	### This method will accept a system size and range of Ec values and generate a random array for purposes of initializing 
	@classmethod
	def generate_Ec_disorder(cls,Lx,Ly,Ecmin,Ecmax):
		return rng.uniform(low=Ecmin,high=Ecmax,size=(Lx,Ly))

	### This method takes the overlap of two wavefunctions 
	### Should have shape of wf = [3,...] where the first dimension is the local Hilbert space index 
	### returns < wf1 | wf2 > 
	@classmethod
	def overlap(cls,wf1,wf2):
		return np.sum( np.conjugate(wf1)*wf2,axis=0)
	
	### This takes the full wavefunction fidelity which is a product over all site sof the overlap
	@classmethod
	def fidelity(cls,wf1,wf2):
		return np.prod( np.sum( np.conjugate(wf1)*wf2,axis=0),axis=(0,1) ) 

	### This evaluates the magnetization <S> on each site 
	### Returns a tensor <S>[c,x,y] with c = 0,1,2,3 the component 
	@classmethod
	def magnetization(cls,wf):
		out = np.zeros((4,*wf.shape[1:]))

		norm = np.real(cls.overlap(wf,wf))

		### We make sure the wavefunction is normalized 
		wf = wf/np.sqrt(norm)

		for i in range(4):
			out[i,...] = np.real( np.sum( np.conjugate(wf) * np.tensordot(cls.spin_one_matrices[i],wf,axes=(1,0)), axis=0) )/norm

		return out

	### This evaluates the charge fluctuations <Sz^2> on each site 
	@classmethod
	def charge_squared(cls,wf):
		norm = cls.overlap(wf,wf)
		### We make sure the wavefunction is normalized 
		wf = wf/np.sqrt(norm)
		out = np.real( np.sum( np.conjugate(wf) * np.tensordot( (cls.spin_one_matrices[3])@(cls.spin_one_matrices[3]), wf,axes=(1,0)),axis=0)/norm)

		return out  


	### Constructor
	def __init__(self,Lx,Ly,Ej,Ec):
		"""Constructor for instance of an xy model with given size and parameters"""
		### We allow for Ec to be an Lx x Ly array of onsite values to include disorder possibility
		self.Lx = Lx 
		self.Ly = Ly 
		self.Ej = Ej 
		self.Ec = Ec 

		### This will hold the mean-field ground state solution 
		self.gs_wf = None 
		self.gs_energy = None 

		### This will be set to the quench function later 
		### for now it is just zero 
		self.quench_function = lambda x : 0.

	### This initializes a wavefunction into the charge insulating mean-field state 
	def initialize_Mott(self):
		wf = np.zeros((3,self.Lx,self.Ly),dtype=complex) ### 3 components for each site, LxxLy sites 

		### According to the parameterization using the spin-one matrices above the charge operator is S^z and thus the [1] component is  the |0> Fock state 
		wf[1,...] = 1.+0.j 

		return wf 

	### This initializes a wavefunction on Lx x Ly grid in the superfluid state with a chosen phase 
	def initialize_SF(self,phase):
		wf = np.zeros((3,self.Lx,self.Ly),dtype=complex) ### 3 components for each site, LxL sites

		wf[0,...] = 0.5*np.exp(1.j*phase)
		wf[1,...] = 0.5*np.sqrt(2.)
		wf[2,...] = 0.5*np.exp(-1.j*phase)

		return wf

	### Evaluates the total energy of an ansatz wavefunction 
	def energy(self,wf):

		### We make sure the wavefunction is normalized 
		norm = self.overlap(wf,wf)
		wf = wf/np.sqrt(np.real(norm))

		charging_energy = np.sum( self.Ec*self.charge_squared(wf) )
		m = self.magnetization(wf)
		Josephson_energy = -0.5*self.Ej*sum([ 
			np.sum( m[1,...]*np.roll(m[1,...], shift=s, axis=[0,1] ) ) + np.sum( m[2,...]*np.roll(m[2,...], shift=s, axis=[0,1] ) )
			for s in [ [0,1], [1,0] ] ])

		return np.real( charging_energy + Josephson_energy )

	### Evaluates the total energy of an ansatz wavefunction 
	### ACCEPTS FLATTENED WAVEFUNCTION FOR OPTIMIZER PURPOSES
	def energy_flat(self,wf_flat):
		wf = wf_flat.reshape((3,self.Lx,self.Ly))
		return self.energy(wf)

	### Evalutes the mean superfluid order parameter 
	@classmethod 
	def SF_OP(cls,wf):
		m = cls.magnetization(wf) ### this is the magnetization at each point in space 

		op = np.mean( m[1,...] + 1.j*m[2,...])

		return op

	### This method will find the ground state for the given system parameters 
	### Optional to accept an initial guess for the wavefunction
	def find_GS(self,wf0=None,**kwargs):
		if wf0 is None:
			wf0 = self.initialize_SF(0.) ### We trial a superfluid ansatz if no other is passed 

		wf0_flat = wf0.flatten()  ### Reshape to correct form for basinhopping

		t0 = time.time()
		res = opt.basinhopping(self.energy_flat,wf0_flat,**kwargs)
		t1 = time.time()

		wf = res.x.reshape((3,self.Lx,self.Ly)) ### reshape and renormalize just to be sure 
		wf = wf/np.sqrt(np.real(self.overlap(wf,wf)))

		self.gs_wf = wf 
		self.gs_energy = res.fun

		return t1 - t0 ### Time taken to compute 

	### This is the equation of motion function in presence of a quench of local vorticity field on plaquette (00) 
	### Requires a quench function to be set, else it will use ground state Hamiltonian 
	### Accepts the flattened wavefunction as an argument
	def eom_quench(self,t,X):
		### First we unflatten X 
		wf = X.reshape((3,self.Lx,self.Ly))

		### We now compute the equation of motion
		### First term is the local charging energy
		### This is -i Ec[x,y] Sz^2 psi[x,y]

		charging_eom = -1.j*np.tensordot( (self.spin_one_matrices[3])@(self.spin_one_matrices[3]), wf,axes=(1,0))*self.Ec

		dXdt = charging_eom.flatten()

		### Now we have the Josephson contributions
		### These are obtained as S.MF
		### Where MF = -0.5*Ej*( m[x+1,y] + m[x-1,y]+m[x,y+1]+m[x,y-1]) 

		m = self.magnetization(wf)

		curie_weiss = -0.5*self.Ej*( np.roll(m,shift=[0,1,0],axis=[0,1,2]) + np.roll(m,shift=[0,-1,0],axis=[0,1,2])+np.roll(m,shift=[0,0,1],axis=[0,1,2]) + np.roll(m,shift=[0,0,-1],axis=[0,1,2]))

		josephson_eom = -1.j*( np.tensordot(self.spin_one_matrices[1],wf,axes=[1,0]) * curie_weiss[1,...] + np.tensordot(self.spin_one_matrices[2] , wf,axes=[1,0])*curie_weiss[2,...] )

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
			out_1 = (2*r - fwd)%np.array([self.Lx,self.Ly])
			out_2 = (2*r - bwd)%np.array([self.Lx,self.Ly])

			### We generate the correct Curie Weiss field including the flux inserted 
			flux_phase = np.exp(1.j*self.quench_function(t)) ### This describes the size and time-dependence of the flux inserted as a function of time 
			complex_curie_weiss = -0.5*self.Ej*flux_phase*(m[1,fwd[0],fwd[1]]+1.j*m[2,fwd[0],fwd[1]]) 
			complex_curie_weiss += -0.5*self.Ej*np.conjugate(flux_phase)*(m[1,bwd[0],bwd[1]]+1.j*m[2,bwd[0],bwd[1]]) 
			complex_curie_weiss += -0.5*self.Ej*(m[1,out_1[0],out_1[1]] + 1.j*m[2,out_1[0],out_1[1]]) 
			complex_curie_weiss += -0.5*self.Ej*(m[1,out_2[0],out_2[1]] + 1.j*m[2,out_2[0],out_2[1]]) 

			### Now we form this in to the appropriate matrix 
			eom_matrix = np.real(complex_curie_weiss)*self.spin_one_matrices[1] + np.imag(complex_curie_weiss)*self.spin_one_matrices[2]

			vorticity_eom[:,x,y] += -1j*np.tensordot(eom_matrix, wf[:,x,y],axes=[1,0])
			josephson_eom[:,x,y] *= 0.

		### We now add the correct eom contributions, which are only nonzero on the plaquette sits, to the josephson terms which are zeroed on the plaquette sites but otherwise correct
		josephson_eom += vorticity_eom

		dXdt += josephson_eom.flatten()

		return dXdt 

	### This computes the wavefunction response to a quench of the local vorticity described by the preset quench function 
	def solve_eom_quench(self,times,wf0=None):
		self.times = times ### Times we want to evaluate wavefunction at 

		### First we find the ground state 
		### We will first find the GS unless we are explicitly passed one as a seed state (useful for symmetry broken GS which may have have orthogonal degenerate GS)
		if wf0 is None:
			wf0 = self.gs_wf

		X0 = wf0.flatten()

		### Now we solve dynamics starting from this, and hopefully find no dynamics in absence of external perturbation
		sol = intg.solve_ivp(self.eom_quench,(self.times[0],self.times[-1]),X0,t_eval=self.times,max_step=0.01)

		### Now we reshape the output and save to the class instance  
		self.wf_vs_t = sol.y.reshape((3,self.Lx,self.Ly,len(sol.t)))

### This class will operate on instances of xy model and coordinate computation of a single echo spectrum 
class xyecho:

	def __init__(self,model,echo_times,flux,sample_times):
		self.model = model ### This is an instance of the xy model class 
		### We assume the ground state has already been obtained 

		### We will consider a generic echo sequence with echo_times = [0,t1,t2,t3,...] and consider the magnetic flux fixed in magnitude but flip sign each echo
		self.echo_times = echo_times 
		self.flux = flux

		self.sample_times = sample_times
		self.ntimes = len(self.sample_times)

		self.wf_shape = self.model.gs_wf.shape

		self.wf = np.zeros_like((*self.wf_shape,self.ntimes))  

	### Now we create the proper quench functions for the different echo sequences 
	def quench_function(self,t):
		bools = t>self.echo_times

		if (bools==False).all() or (bools == True).all():
			return 0. ### We are outside the echo sequence

		else:
			return self.flux*(-1)**int(np.sum(bools)+1)

	### Now we set the dynamics of the model with the given quench function and compute the echo
	def calc_echo(self):
		self.model.quench_function = self.quench_function ### This might be a problem if it modifies the quench function in place for an instance which is used by many different echos and might be better to use inheritance
		self.model.solve_eom_quench(self.sample_times,wf0=self.model.gs_wf)  ### Dynamics. We start from the ground state and sample at the sample times 
		self.wf = self.model.wf_vs_t ### Save the wavefunction dynamics











####################
####### Main #######
####################



















