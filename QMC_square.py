### Code to solve 2+1 d quantum monte carlo XY model on square lattice 
### Jonathan Curtis
### 11/22/24

import numpy as np
from matplotlib import pyplot as plt 
import time 


### We will create a class to handle the simulations 
class QMC:
	### Initialize method
	def __init__(self,EJ,EC,T,L,M):
		### this will initialize a class with the parameters for the model as well as simulation specs
		### EJ = Josephon coupling 
		### EC = Capacitive coupling
		### T = temperature 
		### L = integer physical dimension 
		### M = integer number of imaginary time steps


		### We now initialize the RNG 
		self.rng = np.random.default_rng()

		self.EJ = EJ
		self.EC = EC 
		self.T = T 

		### We round up lattice size to nearest even numbers 
		### In order to implement the proper MCMC sampling we make sure the lattice is bipartite and even size so we can split evenly
		self.L = L if L%2 ==0 else L+1  
		self.M = M if M%2 ==0 else M+1

		### Now we produce the relevant array shape 
		self.shape = (self.L,self.L,self.M)

		### And the relevant time-steps 
		self.beta = 1./self.T
		self.dt = self.beta/M


		### Relevant coupling constants for the 3d model are 
		### This is after (1) trotterizing and (2) making Villain approximation on the time-slices 
		self.Kx = self.EJ*self.dt 
		self.Ky = self.EJ*self.dt 
		self.Kt = 1./(self.EC * self.dt) ### The coupling between neighboring time slices


		### we use an initial condition which is uniform
		self.thetas = np.zeros(self.shape)
		self.thetas = self.rng.random(size =self.shape)*2.*np.pi


	### Modifies the thetas in place one site at a time
	### Works by randomly selecting a site and performign a metropolis-hastings update
	def MCStep(self,thetas):

		### First we randomly propose a site 
		x = self.rng.randint(0,self.L)
		y = self.rng.randint(0,self.L)
		t = self.rng.randint(0,self.M)

		### Now we compute the self-consistent field 
		



	### POSSIBLY DEFECTIVE
	### Modifies the thetas in place 
	def MCStep_old(self,thetas):
		### This implements one time step of the Metropolis update

		### We implement by first breaking in to even and odd sublattices
		### Each of these can be updated independently of the other 
		### We index the sublattice by s = 0,1 for odd or even
		for s in range(2):
			### We compute the self-consistent fields for all sites 
			### Being careful about roll
			nn_indices = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1) ]
			nn_Ks = [ self.Kx,self.Kx,self.Ky,self.Ky,self.Kt,self.Kt ]
			### For each nearest neighbor this is the corresponding spin stiffness in that direction 

			#self_consistent_field = sum([ nn_Ks[nn]*np.exp(-1.j*np.roll(thetas,nn_indices[nn])) for nn in range(len(nn_indices)) ])
			self_consistent_field = sum([ nn_Ks[nn]*np.exp(-1.j*np.roll(thetas, shift=nn_indices[nn], axis=(0,1,2) ) ) for nn in range(len(nn_indices)) ]) 

			new_thetas = self.rng.random(self.shape)*2.*np.pi 
			### These will be the proposal angles for both odd and even, we only need to update SCF after first sweep
			delta_Es = -np.real( ( np.exp(1.j*new_thetas) - np.exp(1.j*thetas) )*self_consistent_field )

			### Now we form an array of accept probabilities
			thresholds = np.exp(-delta_Es)

			### We generate an array of random floats in [0,1] to compare 
			probs = self.rng.random(size=self.shape)

			### This will be one if the entry for that site is accepted and 0 else
			accepts = (probs < thresholds).astype(float)

			### We now elementwise replace the old angles with those that should be updated 
			### We use x ->  x'' = (1-p)*x + p*x' where p is 0,1 depending on whether we accept x' over x (p = 1 is accept x')
			### We mask only the even sublattice and update it 
			### We generate an array mask 
			### sublattice A is if x + y + t is even 
			### This should generate array mask for sublattice A 
			### This piece courtesy of chatGPT
			x = np.arange(self.L)[:,None,None]
			y = np.arange(self.L)[None,:,None]
			t = np.arange(self.M)[None,None,:]
			mask_SL_A = (x+y+t)%2 == 0 

			mask = mask_SL_A if s == 0 else ~mask_SL_A ### for sublattice B we flip the mask
			thetas[mask] += accepts[mask] * (new_thetas[mask] - thetas[mask])	


	##########################
	### SAMPLE OBSERVABLES ###
	##########################

	### This method computes the average free energy density for a particular configuration
	def get_energy_density(self,thetas):
		### This generates a list of nn indices to roll arrays by
		nn_indices = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1) ]
		### For each nearest neighbor this is the corresponding spin stiffness in that direction 
		nn_Ks = [ self.Kx,self.Kx,self.Ky,self.Ky,self.Kt,self.Kt ]

		delta_thetas_nn = [ thetas - np.roll(thetas,nn_index,axis=(0,1,2)) for nn_index in nn_indices ] ### this gives a list of the differences in angles between nearest neighbor sites for each neighbor 
		
		### This computes the change in energies for each site 
		energy_density = 0.

		for nn in range(len(nn_indices)):
			energy_density += - np.mean( nn_Ks[nn]*np.cos(delta_thetas_nn[nn]) )

		return energy_density

	### This computes the vorticity distribution for a given set of angles 
	def get_vorticitiy(self,thetas):
		### This generates a list of nn indices to roll arrays by
		### Note we index the rolls absolutely with respect to the origin of the first array
		### We want A_v = [ sin(theta_{r+x} - theta_r) + sin(theta_{r+x+y} - theta_{r+x} ) + sin(theta_{r+y}-theta_{r+x+y}) + sin(theta_r - theta_{r+y}) ]/4 
		nn_indices = [(-1,0,0),(-1,1,0),(0,-1,0),(0,0,0)]

		vorticity = np.zeros_like(thetas)
		
		for i in range(len(nn_indices)):
			indx1 = nn_indices[i]
			indx2 = nn_indices[i-1]
			vorticity += np.mod( np.roll(thetas,indx1,axis=(0,1,2)) - np.roll(thetas,indx2,axis=(0,1,2)) , 2.*np.pi)

		return vorticity

	### This method computes the mean order parameter as < e^{itheta} > averaged over space and imaginary time
	def get_OP(self,thetas):
		return np.mean(np.exp(1.j*self.thetas))
		
	###########################
	### MC SAMPLING METHODS ###
	###########################

	### Sets the parameters for sampling 
	def set_sampling(self,nburn,nsample,nstep):
	
		### Parameters relevant for MC sampling 
		self.nburn = nburn  ### Number of burn steps. Should be updated at some point to a converging algorithm which runs until converged 
		self.nsample = nsample ### How many samples we want
		self.nstep = nstep ### How many steps between each sample 

		self.theta_samples = np.zeros((self.L,self.L,self.M,self.nsample))
		self.energy_samples = np.zeros(self.nsample)
		self.vort_samples = np.zeros((self.L,self.L,self.M,self.nsample))
		self.OP_samples = np.zeros(self.nsample,dtype=complex)

	### This method implements the burn loop using the single MCStep method for nburn iterations 
	def burn(self):
		for i in range(self.nburn):
			self.MCStep(self.thetas)
			#self.thetas = self.MCStep(self.thetas)

	### We now generate samples and sample the free energy density 
	def sample(self):
		counter = 0 
		while counter < self.nsample:
			### Record the sample
			self.theta_samples[...,counter] = self.thetas
			self.energy_samples[counter] = self.get_energy_density(self.thetas)
			self.vort_samples[...,counter] = self.get_vorticitiy(self.thetas)
			self.OP_samples[counter] = self.get_OP(self.thetas)

			### Now we run for a number of steps 
			for i in range(self.nstep):
				#self.thetas = self.MCStep(self.thetas)
				self.MCStep(self.thetas)

			### Update the counter 
			counter += 1
	
	###########################
	### ANALYSIS OF SAMPLES ###
	###########################



def main():
	t0 = time.time()

	EJ = 1.
	EC = 0.1
	nTs = 10
	Ts = np.linspace(0.05,2.5,nTs)
	L = 10
	M = 10

	nburn = 10000
	nsample = 50
	nstep = 10

	energies = np.zeros(nTs)
	OPs = np.zeros(nTs)

	for i in range(nTs):
		sim = QMC(EJ,EC,Ts[i],L,M)
		sim.set_sampling(nburn,nsample,nstep)

		sim.burn()
		sim.sample()

		energies[i] = np.mean(sim.energy_samples)
		OPs[i] =  np.mean(np.abs(sim.OP_samples) )

	plt.plot(Ts,energies)
	plt.show()
	plt.plot(Ts,OPs)
	plt.show()
	quit()

	plt.plot(sim.energy_samples)
	plt.show()
	plt.plot(np.abs(sim.OP_samples))
	plt.show()
	plt.imshow(sim.vort_samples[:,:,0,-1])
	plt.colorbar()
	plt.show()



	t1 = time.time()
	print("total time: ",t1-t0,"s")




if __name__ == "__main__":
	main()











