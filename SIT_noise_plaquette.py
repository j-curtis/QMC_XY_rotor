### Computes Loschmidt echo for a single square plaquette
### Jonathan Curtis 
### 05/12/25

import numpy as np
import scipy as scp
from scipy import integrate as intg
from scipy import signal 
import time
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as mclr

rng = np.random.default_rng()

### we use matrices [0,1,2,3] to correspond to [identity, paul_x,pauli_y,pauli_z]
spin_one_matrices = [ np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=complex), 1./np.sqrt(2.)*np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]],dtype=complex),1./np.sqrt(2.)*np.array([[0.,-1.j,0.],[1.j,0.,-1.j],[0.,1.j,0.]],dtype=complex),np.array([[1.,0.,0.],[0.,0.,0.],[0.,0.,-1.]],dtype=complex) ]

### Now we form the tensor product matrices across four sites 
### These are labeled 0,1,2,3 starting in upper right and going around counter clockwise
### These are indexed as S[r][c] with c = 0 ,1,2,3 the spin component and r = 0 ,1,2,3 the spatial position 
S = [ [ np.kron(spin_one_matrices[0],np.kron(spin_one_matrices[0],np.kron(spin_one_matrices[0],spin_one_matrices[c]) ) ) for c in range(4) ] ,
[ np.kron(spin_one_matrices[0],np.kron(spin_one_matrices[0],np.kron(spin_one_matrices[c],spin_one_matrices[0]) ) ) for c in range(4) ] ,
[ np.kron(spin_one_matrices[0],np.kron(spin_one_matrices[c],np.kron(spin_one_matrices[0],spin_one_matrices[0]) ) ) for c in range(4) ] ,
[ np.kron(spin_one_matrices[c],np.kron(spin_one_matrices[0],np.kron(spin_one_matrices[0],spin_one_matrices[0]) ) ) for c in range(4) ] ]


dim = (S[0][0]).shape[0] ### Judge from the size of the matrix which should be 81


### This is the hamiltonian for a plaquette with a flux phi threaded 
def H(Ec,Ej,flux):
	Hc = Ec*sum([ S[r][3]@S[r][3] for r in range(4) ])

	hopping_phase = np.exp(-1.j*flux)
	### Hopping out of the page 
	Hccw = -0.5*Ej*hopping_phase*sum([ (S[r][1]+1.j*S[r][2])@(S[r-1][1]-1.j*S[r-1][2]) for r in range(4) ])
	Hcw = -0.5*Ej*np.conjugate(hopping_phase)*sum([ (S[r][1]-1.j*S[r][2])@(S[r-1][1]+1.j*S[r-1][2]) for r in range(4) ])


	return Hc + Hccw + Hcw

def find_GS(Ec,Ej):
	h = H(Ec,Ej,0.)

	es,psis = np.linalg.eigh(h)

	return es[0],psis[:,0]

### Evolution function under Hamiltonian h 
def eom_h(t,X,h):
	return -1.j*h@X


### This method evolves from the ground state under a Hamiltonian with given Ej and Ec and a time-dependent flux which switches sign at indicated times 
def evolve_quench(Ec,Ej,echo_times,flux,sample_times):

	###  First we define the quench function 
	def quench_function(t):
		bools = t>echo_times

		if (bools==False).all() or (bools == True).all():
			return 0. ### We are outside the echo sequence

		else:
			return flux*(-1)**int(np.sum(bools)+1)

	### now we generate the Hamiltonian 
	def eom_quench(t,X): 
		return -1.j*H(Ec,Ej,quench_function(t))@X

	### Now we find the ground state 
	e_gs, psi_gs = find_GS(Ec,Ej)

	### Next we evolve under the equations of motion 
	sol = intg.solve_ivp(eom_quench,(sample_times[0],sample_times[-1]),psi_gs,t_eval=sample_times,max_step = 0.05)
	return sol.y


def Ramsey_echo(tRs,lps,Ec,Ej):
	ntR = len(tRs)
	nlp = len(lps)

	e_gs, psi_gs = find_GS(Ec,Ej)

	echos = np.zeros((nlp,ntR),dtype=complex)
	psi_t = np.zeros((dim,ntR),dtype=complex)

	for i in range(nlp):
		h = H(Ec,Ej,lps[i])

		psi_t = (intg.solve_ivp(eom_h,(tRs[0],tRs[-1]),psi_gs,args=(h,),t_eval=tRs) ).y

		for j in range(ntR):
			echos[i,j] = np.conjugate(psi_gs)@(psi_t[:,j])

	return psi_t

### Double check this can be written as overlap of two different Ramsey echos 
### Seems like this is missing an extra period of evolution 
def Hahn_echo(tHs,lps,Ec,Ej):
	ntH = len(tHs)
	nlp = len(lps)

	e_gs, psi_gs = find_GS(Ec,Ej)

	echos = np.zeros((nlp,ntH),dtype=complex)

	for i in range(nlp):
		h1 = H(Ec,Ej,lps[i])
		h2 = H(Ec,Ej,-lps[i])

		psi_1 = (intg.solve_ivp(eom_h,(tHs[0],tHs[-1]),psi_gs,args=(h1,),t_eval=tHs) ).y
		psi_2 = (intg.solve_ivp(eom_h,(tHs[0],tHs[-1]),psi_gs,args=(h2,),t_eval=tHs) ).y

		for j in range(ntH):
			echos[i,j] = np.conjugate(psi_2[:,j])@(psi_1[:,j])

	return psi_1, psi_2
	#return echos




def main():
	Ec = 1.
	Ej = 0.0

	ntR = 500
	nlp = 45
	tRs_short = np.linspace(0.,25.,ntR)
	tRs_long = np.linspace(0.,50.,ntR)
	tHs = np.linspace(0.,10.,ntR)
	lps = np.linspace(0.,1.,nlp)

	Ramsey_echos_short = Ramsey_echo(tRs_short,lps,Ec,Ej)
	Ramsey_echos_long = Ramsey_echo(tRs_long,lps,Ec,Ej)
	Hahn_echos = Hahn_echo(tHs,lps,Ec,Ej)


	nonGaussian_echos = np.zeros_like(Hahn_echos)

	nonGaussian_echos = np.log(Hahn_echos) + np.log(Ramsey_echos_long) - 4.*np.log(Ramsey_echos_short)


	plt.imshow(np.real(Ramsey_echos_short),origin='lower',extent=[tRs_short[0],tRs_short[-1],lps[0],lps[-1]],cmap='coolwarm')
	plt.xlabel(r'$\tau_R E_c$')
	plt.ylabel(r'$\lambda_P/E_c$')
	plt.colorbar()
	plt.show()


	plt.imshow(np.real(Ramsey_echos_long),origin='lower',extent=[tRs_long[0],tRs_long[-1],lps[0],lps[-1]],cmap='coolwarm')
	plt.xlabel(r'$\tau_R E_c$')
	plt.ylabel(r'$\lambda_P/E_c$')
	plt.colorbar()
	plt.show()

	plt.imshow(np.real(Hahn_echos),origin='lower',extent=[tHs[0],tHs[-1],lps[0],lps[-1]],cmap='coolwarm')
	plt.xlabel(r'$\tau_H E_c$')
	plt.ylabel(r'$\lambda_P/E_c$')
	plt.colorbar()
	plt.show()

	plt.imshow(np.real(nonGaussian_echos),origin='lower',extent=[tHs[0],tHs[-1],lps[0],lps[-1]],cmap='coolwarm')
	plt.xlabel(r'$\tau_{\rm NG} E_c$')
	plt.ylabel(r'$\lambda_P/E_c$')
	plt.colorbar()
	plt.show()

	plt.imshow(np.imag(nonGaussian_echos),origin='lower',extent=[tHs[0],tHs[-1],lps[0],lps[-1]],cmap='coolwarm')
	plt.xlabel(r'$\tau_{\rm NG} E_c$')
	plt.ylabel(r'$\lambda_P/E_c$')
	plt.colorbar()
	plt.show()

if __name__ == "__main__":
	main()




