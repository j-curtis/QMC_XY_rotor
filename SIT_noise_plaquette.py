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

def eom_f(t,X,h):
	return -1.j*h@X

def main():
	Ec = 1.
	Ej = 0.0

	e_gs, psi_gs = find_GS(Ec,Ej)



	ntR = 300
	nlp = 300
	tRs = np.linspace(0.,20.,ntR)
	lps = np.linspace(0.,10.,nlp)

	echos = np.zeros((nlp,ntR),dtype=complex)

	for i in range(nlp):
		h = H(Ec,Ej,lps[i])

		psi_t = (intg.solve_ivp(eom_f,(tRs[0],tRs[-1]),psi_gs,args=(h,),t_eval=tRs) ).y

		for j in range(ntR):
			echos[i,j] = np.conjugate(psi_gs)@(psi_t[:,j])


	plt.imshow(np.real(echos),origin='lower',extent=[tRs[0],tRs[-1],lps[0],lps[-1]],cmap='coolwarm')
	plt.xlabel(r'$\tau_R E_c$')
	plt.ylabel(r'$\lambda_P/E_c$')
	plt.colorbar()
	plt.show()


if __name__ == "__main__":
	main()




