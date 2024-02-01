## @package Lanczos This package is a very simple and straight-forward implementation of the
# Lanczos algorithm for sparse matrices.
#
# Usage: Create a LanczosSolver object with the desired properties (max. number of
# iterations, desired precision, invariant subspace detection threshold etc).
#
# Then either call the 'lanczos' method to compute ground state energy and ground
# state vector, or call the first lanczos pass individually and then explicitly
# diagonalize the resulting matrix to obtain spectral weights.'''

import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_la
import logging

## Return the adjunct of a matrix, which really is just
#  `conjugate(transpose(x))`.
def adj(x):
    return np.conjugate(np.transpose(x))

## This class represents an instance of a lanzcos solver with its own
#  set of parameters and intermediate states.
class LanczosSolver:

    ## Initialize the lanczos solver with its parameters.
    #
    #  @param kwargs The keywords to be used:
    #         - maxiter: Maximum number of iterations
    #         - precision: Desired tolerance for the lanczos
    #         - cond: If it's not "PRECISION", always run up to `maxiter`.
    #         - eps: Tolerance for detecting loss of orthogonality
    def __init__(self, **kwargs):
        self.maxiter = kwargs.get('maxiter')
        self.precision = kwargs.get('precision')
        self.cond = kwargs.get('cond')
        self.eps = kwargs.get('eps')
        
        self.alpha = np.empty(self.maxiter,dtype=complex)
        self.beta = np.empty(self.maxiter - 1,dtype=complex)
        self.m = None
        self.gse = None
        self.passed_first = False

    ## Clear all computed quantities but keep the parameters
    def reset(self):
        '''Clears all computed quantities but keeps the parameters'''
        self.alpha = np.empty(self.maxiter,dtype=complex)
        self.beta = np.empty(self.maxiter - 1,dtype=complex)
        self.m = None
        self.gse = None
        self.passed_first = False
        
    ## The main function of this class, taking care of a full lanczos run.
    #
    #  @param kwargs Set of keyword arguments. Use
    #                - `x0`: Starting vector. Use something randomly initialized
    #                - `scratch`: Pre-allocated space for a scratch vector
    #                - `y`: Allocated space where the ground state vector will be stored
    #                - `H`: A sparse matrix representing the Hamiltonian
    #  @return ev The ground state eigenvalue of the Hamiltonian. Also returns the ground state
    #             vector in the numpy array `y`.
    def lanczos(self,**kwargs):
        start_vector = kwargs.get('x0')
        scratch_vector = kwargs.get('scratch')
        ground_state = kwargs.get('y')
        H = kwargs.get('H')

        if not abs(linalg.norm(start_vector) - 1) < self.eps:
            start_vector /= linalg.norm(start_vector)

        start_vector_copy = np.copy(start_vector)
        ev = self.lanczos_pass(mode = 'FIRST', x0 = start_vector_copy,
                          scratch = scratch_vector, y = ground_state, H = H)
        self.lanczos_pass(mode = 'SECOND', x0 = start_vector,
                     scratch = scratch_vector, y = ground_state, H = H)
        return ev

    ## Only perform the first pass of the Lanczos algorithm
    def first_pass(self,**kwargs):
        kwargs['mode'] = 'FIRST'
        self.lanczos_pass(**kwargs)
        
    ## Perform a pass (first or second) of the Lanczos algorithm
    def lanczos_pass(self,**kwargs):
        mode = kwargs.get('mode',"FIRST")
        start_vector = kwargs.get('x0')
        scratch_vector = kwargs.get('scratch')
        ground_state = kwargs.get('y')
        H = kwargs.get('H')

        b = start_vector
        r = ground_state
        q = scratch_vector

        y = None

        #Don't allow second pass to be called if first pass wasn't.
        assert mode == 'FIRST' or self.passed_first
        
        if mode == 'SECOND':
            tmp = np.diag(self.alpha[0:self.m]) + (np.diag(self.beta[0:self.m-1],k=1) +
                                                     np.diag(self.beta[0:self.m-1],k=-1) )
            V,D = linalg.eigh(tmp)
            indices = np.argsort(V)
            y = D[:,indices[0]]
            norm = np.conjugate(y).dot(y)
            y = y / np.sqrt(norm)
        q[:] = 0
        if mode == 'SECOND':
            r[:] = 0

        j = 0
        gse_old = 0
        gse_new = 0
        while (mode == 'FIRST' and j < self.maxiter) or (mode == 'SECOND' and j < self.m):
            if not j == 0:
                b *= -self.beta[j-1]
                q *= (1.0/self.beta[j-1])
                tmp = b
                b = q
                q = tmp

            if mode == 'SECOND':
                r += (y[j] * b)

            q += H.dot(b)

            if mode == 'FIRST':
                self.alpha[j] = adj(b).dot(q)

            q += (-self.alpha[j] * b)

            if j < self.maxiter - 1 and mode == 'FIRST':
                self.beta[j] = linalg.norm(q)
                if mode == 'FIRST' and self.beta[j] < self.eps:
                    print ("Beta very small: Invariant sub-space reached after ", j, " iterations")
                    break
            j += 1

            #Now diagonalize the tridiagonal symmetric matrix defined
            if (self.cond == 'PRECISION' or j == self.maxiter) and mode == 'FIRST':
                if j == self.maxiter and self.cond == 'PRECISION':
                    logging.warning("Warning: Max number of iterations reached")
                if j > 0:
                    gse_old = gse_new

                tmp = np.diag(self.alpha[0:j]) + (np.diag(self.beta[0:j-1],k=1) +
                                                        np.diag(self.beta[0:j-1],k=-1) )

                V,D = linalg.eigh(tmp)
                gse_new = V.min()
                ev = gse_new
                self.gse = gse_new

                if j > 1:
                    self.relative_error = abs((gse_old - gse_new) / gse_new)
                    logging.debug("error: %4.4g" % self.relative_error)
                    if self.relative_error < self.precision:
                        break

        self.m = j
        self.passed_first = True
        
        return self.gse

    
    ## Diagonalize the tridiagonal matrix `T` generated in the first pass of the Lanczos.
    def lanczos_diag_T(self):
        '''After completing the first pass, we have the tridiagonal matrix "T"
        the represents the Hamiltonian H in the Krylov subspace. This method
        diagonalizes T, returning the eigenvalues in the first and the
        eigenvectors in the second argument'''
        m = self.m
        assert not m == 0
        tmp = np.diag(self.alpha[0:m-1]) + (np.diag(self.beta[0:m-2],k=1) +
                                            np.diag(self.beta[0:m-2],k=-1) )
        V, D = linalg.eigh(tmp)
        #Sort
        indices = np.argsort(V)
        V = V[indices]
        D = D[:,indices]
        return V,D

    def lanczos_invert_T(self,E,eta):
        m = self.m
        assert not m == 0
        tmp = (np.diag(E + 1j*eta - self.alpha[0:m-1]) 
               - np.diag(self.beta[0:m-2],k=1) 
               - np.diag(self.beta[0:m-2],k=-1) )
        
        #tmp_sparse = sparse.csr_matrix(tmp)
        b = np.zeros(m-1,dtype=complex)
        b[0] = 1
        x = np.linalg.solve(tmp, b)
#        x = sp_la.spsolve(tmp_sparse, b)
        return x[0]
