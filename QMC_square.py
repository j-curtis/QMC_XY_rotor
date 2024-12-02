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
		self.L = L 
		self.M = M 

		### Now we produce the relevant array shape 
		self.shape = (L,L,M)

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

	### This method will implement a single time-step of the Metropolis Hastings sampling for us
	def MCStep(self,thetas):
		
		### thetas is passed theta configuration
		### This will return a new generated theta config
		### One time step will be one sweep through the lattice 
		out = np.copy(thetas) 

		### This generates a list of nn indices to roll arrays by
		nn_indices = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1) ]
		### For each nearest neighbor this is the corresponding spin stiffness in that direction 
		nn_Ks = [ self.Kx,self.Kx,self.Ky,self.Ky,self.Kt,self.Kt ]

		delta_thetas_nn = [ thetas - np.roll(thetas,nn_index) for nn_index in nn_indices ] ### this gives a list of the differences in angles between nearest neighbor sites for each neighbor 

		new_thetas = self.rng.random(size=self.shape)*2.*np.pi

		delta_thetas_nn_random = [ new_thetas - np.roll(thetas,nn_index) for nn_index in nn_indices ] ### Sample for the proposed updated thetas

		### This computes the change in energies for each site 
		delta_Es = np.zeros_like(thetas)

		for nn in range(len(nn_indices)):
			delta_Es += -nn_Ks[nn]*(np.cos(delta_thetas_nn_random[nn]) - np.cos(delta_thetas_nn[nn]) ) 

		### Now we form an array of accept probabilities
		thresholds = np.exp(-delta_Es)

		### We generate an array of random floats in [0,1] to compare 
		probs = self.rng.random(size=self.shape)

		### This will be one if the entry for that site is accepted and 0 else
		accepts = (probs < thresholds).astype(float)

		### We now elementwise replace the old angles with those that should be updated 
		### We use x ->  x'' = (1-p)*x + p*x' where p is 0,1 depending on whether we accept x' over x (p = 1 is accept x')
		out = out + accepts * (new_thetas - out)

		### We now mod back to 2pi 
		out = np.mod(out,2.*np.pi)

		return out 

	### This method computes the average free energy density for a particular configuration
	def get_energy_density(self,thetas):
		### This generates a list of nn indices to roll arrays by
		nn_indices = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1) ]
		### For each nearest neighbor this is the corresponding spin stiffness in that direction 
		nn_Ks = [ self.Kx,self.Kx,self.Ky,self.Ky,self.Kt,self.Kt ]

		delta_thetas_nn = [ thetas - np.roll(thetas,nn_index) for nn_index in nn_indices ] ### this gives a list of the differences in angles between nearest neighbor sites for each neighbor 
		
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
		nn_indices = [(1,0,0),(1,1,0),(0,1,0),(0,0,0)]

		vorticity = np.zeros_like(thetas)
		
		for i in range(len(nn_indices)):
			indx1 = nn_indices[i]
			indx2 = nn_indices[i-1]
			vorticity += np.sin( np.roll(thetas,indx1) - np.roll(thetas,indx2) )/4.

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
			self.thetas = self.MCStep(self.thetas)

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
				self.thetas = self.MCStep(self.thetas)

			### Update the counter 
			counter += 1
	
	###########################
	### ANALYSIS OF SAMPLES ###
	###########################



def main():
	t0 = time.time()

	EJ = 1.
	EC = 0.2
	T = 3.
	L = 6
	M = 30

	sim = QMC(EJ,EC,T,L,M)
	print(sim.Kx,sim.Ky,sim.Kt)

	nburn = 1000000000
	nsample = 20
	nstep = 1000

	sim.set_sampling(nburn,nsample,nstep)

	sim.burn()
	sim.sample()

	plt.plot(sim.energy_samples)
	plt.show()
	plt.plot(np.abs(sim.OP_samples))
	plt.show()
	plt.plot(sim.vort_samples[0,0,0,:])
	plt.show()
	plt.imshow(sim.theta_samples[:,:,0,0],origin='lower')
	plt.show()
	plt.imshow(np.mean(sim.vort_samples[:,:,0,:],axis=-1),origin='lower',cmap='coolwarm')
	plt.colorbar()
	plt.show()



	t1 = time.time()
	print("total time: ",t1-t0,"s")




if __name__ == "__main__":
	main()











