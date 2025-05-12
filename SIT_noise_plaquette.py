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
dim = 3**4

### we use matrices [0,1,2,3] to correspond to [identity, paul_x,pauli_y,pauli_z]
spin_one_matrices = [ np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=complex), 1./np.sqrt(2.)*np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]],dtype=complex),1./np.sqrt(2.)*np.array([[0.,-1.j,0.],[1.j,0.,-1.j],[0.,1.j,0.]],dtype=complex),np.array([[1.,0.,0.],[0.,0.,0.],[0.,0.,-1.]],dtype=complex) ]

### Now we form the tensor product matrices across four sites 
### These are labeled 0,1,2,3 starting in upper right and going around counter clockwise
### These are indexed as S[r][c] with c = 0 ,1,2,3 the spin component and r = 0 ,1,2,3 the spatial position 
S = [ [ np.kron(spin_one_matrices[0],np.kron(spin_one_matrices[0],np.kron(spin_one_matrices[0],spin_one_matrices[c]) ) ) for c in range(4) ] ,
[ np.kron(spin_one_matrices[0],np.kron(spin_one_matrices[0],np.kron(spin_one_matrices[c],spin_one_matrices[0]) ) ) for c in range(4) ] ,
[ np.kron(spin_one_matrices[0],np.kron(spin_one_matrices[c],np.kron(spin_one_matrices[0],spin_one_matrices[0]) ) ) for c in range(4) ] ,
[ np.kron(spin_one_matrices[c],np.kron(spin_one_matrices[0],np.kron(spin_one_matrices[0],spin_one_matrices[0]) ) ) for c in range(4) ] ]

def H(Ec,Ej,lp):
	Hc = Ec*sum([ S[r][3]@S[r][3] for r in range(4) ])
	Hj = -Ej*sum([ sum([ S[r-1][c]@S[r][c] for c in [1,2] ]) for r in range(4) ])

	Vp = -0.25*lp*sum([ S[r][2]@S[r-1][1]  - S[r][1]@S[r-1][2]  for r in range(4) ])

	return Hc + Hj + Vp

def find_GS(Ec,Ej):
	h = H(Ec,Ej,0.)

	es,psis = np.linalg.eigh(h)

	return es[0],psis[0,:]

### Evolution function under Hamiltonian h 
def eom_h(t,X,h):
	return -1.j*h@X


def Ramsey_echo(tRs,lps,Ec,Ej):
	ntR = len(tRs)
	nlp = len(lps)

	e_gs, psi_gs = find_GS(Ec,Ej)

	echos = np.zeros((nlp,ntR),dtype=complex)

	for i in range(nlp):
		h = H(Ec,Ej,lps[i])

		psi_t = (intg.solve_ivp(eom_h,(tRs[0],tRs[-1]),psi_gs,args=(h,),t_eval=tRs) ).y

		for j in range(ntR):
			echos[i,j] = np.conjugate(psi_gs)@(psi_t[:,j])

	return echos

### Double check this can be written as overlap of two Ramsey echos 
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

	return echos




def main():
	Ec = 1.
	Ej = 0.0

	ntR = 100
	nlp = 100
	tRs_short = np.linspace(0.,10.,ntR)
	tRs_long = np.linspace(0.,20.,ntR)
	tHs = np.linspace(0.,10.,ntR)
	lps = np.linspace(0.,10.,nlp)

	Ramsey_echos_short = Ramsey_echo(tRs_short,lps,Ec,Ej)
	Ramsey_echos_long = Ramsey_echo(tRs_long,lps,Ec,Ej)
	Hahn_echos = Hahn_echo(tHs,lps,Ec,Ej)


	nonGaussian_echos = np.zeros_like(Hahn_echos)

	nonGaussian_echos = Hahn_echos*Ramsey_echos_long/(Ramsey_echos_short**4)


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

	plt.imshow(np.real(np.log(nonGaussian_echos)),origin='lower',extent=[tHs[0],tHs[-1],lps[0],lps[-1]],cmap='coolwarm')
	plt.xlabel(r'$\tau_{\rm NG} E_c$')
	plt.ylabel(r'$\lambda_P/E_c$')
	plt.colorbar()
	plt.show()

if __name__ == "__main__":
	main()




