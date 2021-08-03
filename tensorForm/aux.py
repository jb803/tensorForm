"""Module for eigenvalue problems.

We have an abstract base-class representing an eigensolver. This has a number of functions which must be overloaded"""

from abc import ABC,abstractmethod

from petsc4py import PETSc 
from slepc4py import SLEPc

import dolfin as dolf

import time

import numpy as np

import os

import gc

class EigenError(Exception):
    """Error class for errors arising in eigenvalue problems"""
    def __init__(self,message):
        self.message=message

class Eigensolver(ABC):
    """Abstract base class for an eigensolver.

    This abstract class implements some of the required grunt work in :py:meth:`_buildEPS`. The methods that need to
    be overloaded are:

    * :py:meth:`build` - builsd the matrices and SLEPc.EPS object needed for the eigensolve.
    * :py:meth:`validateEigenpair` - validates an eigenapair (some eigensolvers will introduce false solutions)
    * :py:meth:`getEigenpair` - returns an eigenpair.
    """

    def __init__(self):
        super().__init__()
        self._defaultLogTime = 10.

        self.built = False
        """Flag to indicate whether the matrices have been built"""

    def _resetTimes(self):
        """Resets internal timers for the stop and logging during eps solves"""
        self._lastLog   = time.time()
        self._startTime = time.time()

    def _buildEPS(self,*ops):
        """
        Performs the grunt work for build a SLEPC eps object with PETSC matrices as the operators.

        This is called by classes inheriting from :py:class:`Eigensolver`. 

        :params list(PETSc.Mat) ops: set of operator matrices to pass to the eigensolver
        """
        eps = SLEPc.EPS().create()      #Generate EPS
        st = eps.getST()
        st.setType('sinvert')           #shift invert

        #Set up the linear solver
        ksp = st.getKSP()
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')

        #Set the operators
        eps.setOperators(*ops)

        eps.setStoppingTest(self._eigensolveStop)

        self.eps = eps

    def _eigensolveStop(self, solver, its, max_it, nconv, nev):
        currTime    = time.time()
        solveTime   = currTime - self._startTime

        if currTime - self._lastLog > self._defaultLogTime:
            self._lastLog = currTime

        if nconv >= nev:
            return SLEPc.EPS.ConvergedReason.CONVERGED_TOL

        if self._maxSolveTime is not None and solveTime > self._maxSolveTime and its > 8:
            return SLEPc.EPS.ConvergedReason.CONVERGED_USER

        if its > max_it:
            return SLEPc.EPS.ConvergedReason.DIVERGED_ITS

        return SLEPc.EPS.ConvergedReason.ITERATING


    @abstractmethod
    def build(self):
        """Builds any matrices required and creates the eigensolver.

        Must be overloaded by any class inheriting this.
        """
        pass

    def solve(self,nEigen, maxSolveTime = 240):
        """Solves for eigenvalues.

        This is performs the grunt work of performing an eigensolve through the SLEPC interface. It uses the :py:attr:`built`
        flag to determing whether or not to call :py:meth:`build` . 

        :param int nEigen: number of eigenvalues to look for
        :return: number of eigenvalues found
        :rtype: int
        """
        #We make sure we have built all the required matrices and solvers

        if not self.built:
            self.build()
        
        #TODO: maybe check if the build flag has raised?
        #Set how many eigenvalues to look for
        self.eps.setDimensions(nEigen,SLEPc.DECIDE)

        self._resetTimes()
        self._maxSolveTime = maxSolveTime

        gc.collect(2) #We should garbage collect before solving #TODO - need to sort out when and how this happens. May do it too oftern
        self.eps.solve()


        self.validatedEigenpairs = []
        self.storedEigenpairs = []

        self.solved = True

        return self.eps.getConverged()
        
    def nConverged(self):
        """Returns number of eigenvalues converged.

        :return: number of eigenvalues converged
        :rtype: int
        """
        return self.eps.getConverged()



    @abstractmethod
    def getEigenpair(self,i,absSpectra=True):
        """Retrieves eigenpair.

        Must be overloaded by any class inheriting this.

        :param int i: which eigenpair to return.
        :param bool absSpectra: (optional) for use if a shift is involved. If true then the eigenvalue is the absolute position in the spectra as opposed to the position relative to the shift
        """
        pass


#COMPLEX EIGENVALUE PROBLEM#
class ComplexEigensolver(Eigensolver):
    """This is an eigensolver that takes the real representation of a complex eigenvalue and allows it to be solver using
    real PETSc. 

    .. todo::

        With no shift then this struggles validating the eigenvalues, it should always be run with a shift

    Because all the matrices are real, it morphs an NxN problem into a 2Nx2N problem. See :ref:`complex-eigensolver-label` for
    details.

    :param dolfin.PETScMatrix AReal: The real part of the A operator
    :param dolfin.PETScMatrix AImag: The imaginary part of the A operator
    :param dolfin.PETScMatrix B: (optional) the B operator of a generalised eigenvalue problem
    :param str mode: (optional) whether to use direct mode (right eigenvectors, default) or adjoint mode (left eigenvectors)
    """
    def __init__(self, AReal, AImag, B=None, mode = None):
        super().__init__()

        #TODO: implement dealing with no B matrix
        if B == None:
            raise EigenError('Not yet implemented ComplexEigensolver without B specified')

        self.AReal  = AReal
        self.AImag  = AImag
        self.B      = B 

        self.shift = 0.
        self.solved = False

        self.validatedEigenpairs = []
        self.storedEigenpairs = []

        if mode is None:
            self.mode = 'direct'
        else:
            self.mode = mode


    def build(self):
        """Builds the necessary matrices needed for SLEPc eigensolver."""

        self.solved = False  #we make sure we clear the solved flag
        self._validated = False
        self.validatedEigenpairs = []
        self.storedEigenpairs = []

        A00 = self.AReal.mat().copy()

        if self.AImag != 0:
            A10 = self.AImag.mat().copy()

        B00 = self.B.mat().copy()
        
        dimBlock = A00.getSize()

        #The case of zero real part of the shift causes problems and a real eigenvalue problem, leads to the case where ghosts can't be differentiated
        if self.shift.imag == 0 and self.AImag == 0:
            #We have a special case where we have a purely real system. The splitting into 2Nx2N makes no sense and so we
            #produce a reduced system
            A00.axpy(-self.shift.real, B00)
            self.AA = A00
            self.BB = B00


        else:
            #We have to produce a split system

            #We check all the matrices we've been given are compatible
            if dimBlock != B00.getSize():
                raise EigenError('Matrices A and B must agree in dimension')
            if self.AImag != 0 and dimBlock != A10.getSize():
                raise EigenError('Real part and complex part of A must agree')

            dummy = PETSc.Mat() #dummy matrix for building the block system

            comm = A00.getComm() #We use the same mpi communicator that dolfin was using
            
            #We construct the augmented A matrix
            A11 = A00.copy() #get base copy for A11
            A00.axpy(-self.shift.real,B00)
            A11.axpy(-self.shift.real,B00)

            if self.AImag == 0:
                A01 = B00.copy()
                A01.scale(self.shift.imag)

                A10 = B00.copy()
                A10.scale(-self.shift.imag)

            else:
                A01 = B00.copy()
                A01.scale(self.shift.imag)
                A01.axpy(-1., A10)

                A10.axpy(-self.shift.imag, B00)

            #We construct the augmented B matrix
            BB  = PETSc.Mat().createNest([[B00  ,dummy],
                                          [dummy,  B00]] ,comm=comm)

            AA = PETSc.Mat().createNest([[A00, A01],
                                         [A10, A11]],comm=comm)

            #We now convert the matrices to SEQ AIJ form
            AA.convert(mat_type=PETSc.Mat.Type.AIJ,out = AA)
            BB.convert(mat_type=PETSc.Mat.Type.AIJ,out = BB)

            self.AA = AA
            self.BB = BB

            self.indicesReal = PETSc.IS().createBlock(dimBlock[0],0)
            self.indicesImag = PETSc.IS().createBlock(dimBlock[0],1)

        #store dimensions
        self.dimBlock = dimBlock
        self.dim      = self.AA.getSize()

        #We generate the eps obejct
        if self.mode == 'adjoint':
            self.AA.transpose(out=self.AA)
            self.BB.transpose(out=self.BB)
        self._buildEPS(self.AA,self.BB)

        self.built=True

    def applyShift(self,shift):
        """Applies a complex shift.

        This clears the built flag and so the matrices will be rebuilt for the next solve.

        :param complex shift: complex shift to apply
        """
        self.shift = shift
        self.built = False

    def _reconstructEigenvector(self,rx,cx):
        """Reconstructs the original real and imaginary parts of the eigenvector.

        Reconstruction is required because the spltting into 2Nx2N redistributes the eigenvector
        into real and imaginary parts.

        :param PETSC.Vec rx: PETSc vector representing the real part of the eigenvector
        :param PETSC.Vec cx: PETSc vector representing the complex part of the eigenvector
        :return: tuple of PETSC.Vec of the reconstructed real and imaginary parts of the eigenvector
        """
        #We generate the index sets for the block which represents the real component and the block which represents the complex component
        
        if self.shift.imag != 0:

            r0 = PETSc.Vec().createSeq(self.dimBlock[0]);  rx.getSubVector(self.indicesReal, subvec=r0)
            r1 = PETSc.Vec().createSeq(self.dimBlock[0]);  rx.getSubVector(self.indicesImag, subvec=r1)
            c0 = PETSc.Vec().createSeq(self.dimBlock[0]);  cx.getSubVector(self.indicesReal, subvec=c0)
            c1 = PETSc.Vec().createSeq(self.dimBlock[0]);  cx.getSubVector(self.indicesImag, subvec=c1)

            r0.axpy(-1.,c1)
            r1.axpy( 1.,c0)

            return r0, r1
        else:
            return rx,cx

    def validateEigenpairs(self, store=False, sTol = 1e-3, rTol = 1e3):
        """Validates the converged eigenpairs.

        This is done to eliminate the additional 'ghost' solutions that arise from the 2Nx2N real representation of a
        complex matrix. The way this works is that we know the ghost solutions appear as eigenvalues that are exactly mirrored
        about the sinvert point

        :param bool store: (optional) whether to store the validated eigenpairs for future use
        :param float sTol: (optional) spectral tolerance for checking mirrored eigenvalues. Default is 1e-3
        :param flaot rTol: (optional) relative tolerance for checking which mode is the ghost and which is the real eigenvector. Default is 1e3. 

        :return: list of validated eigenpairs
        :rtype: bool

        """
        if not self.solved:
            raise EigenError('Must call solve() before validating eigenpairs')

        #we clear the 


        #We identify the pairs
        evList      = []
        procEvs     = []
        evGroups    = []
        validPairs  = []

        nEv = self.eps.getConverged()

        rx = PETSc.Vec().createSeq(self.dim[0])
        cx = PETSc.Vec().createSeq(self.dim[0])

        counter = 0

        if self.shift.imag == 0:

            self.validatedEigenpairs = [i for i in range(nEv)]
            self.storedEigenpairs    = [[self.eps.getEigenvalue(i)] for i in range(nEv)]

            if store:
                for i in range(nEv):
                    self.eps.getEigenpair(i, rx, cx)
                    self.storedEigenpairs[i].extend([rx.copy(), cx.copy()])

        else:
            #The imaginary shift means we get possible ghost solutions
            #We get all the eigenvalues
            for i in range(nEv):
                evList.append([self.eps.getEigenvalue(i), i])

            #We organise them into potential ghost/no ghost pairs
            for i in range(nEv):
                #We locate groups
                if i in procEvs: continue

                evI = evList[i][0]

                newGroup = []

                newGroup.append(evList[i])
                procEvs.append(i)

                for j in range(i+1, nEv):
                    #in the relative spectrum frame these ghost pairs should be complex conjugates of each other
                    if j in procEvs: continue

                    evJ = evList[j][0]

                    if np.abs(evI.real - evJ.real) < sTol and np.abs(evI.imag + evJ.imag) < sTol:
                        #this is a candidate for being part of a ghost/no ghost pair
                        newGroup.append(evList[j])
                        procEvs.append(j)

                evGroups.append(newGroup)


            for evGroup in evGroups:
                #We process the groups looking for which of the pair has the largest norm
                

                if len(evGroup) > 2:
                    continue
                    #raise Exception('Cannot handle ev groups of more than 2 members')
                elif len(evGroup) == 1:

                    pair = [evGroup[0][0]]

                    if store:

                        #self.eps.getEigenpair(evGroup[0][1], rx, cx)
                        norm, real, imag = self._eigenpairNorm(evGroup[0][1])
                        pair.extend([real, imag])

                    self.validatedEigenpairs.append(evGroup[0][1])
                    self.storedEigenpairs.append(pair)

     
                else:
                    #We have a two group. We need to check the relative magnitudes
                    norm0, rx0, cx0 = self._eigenpairNorm(evGroup[0][1])
                    norm1, rx1, cx1 = self._eigenpairNorm(evGroup[1][1])


                    if norm0 > norm1:
                        index = evGroup[0][1]
                        ev    = evGroup[0][0]
                        rx    = rx0
                        cx    = cx0

                    else:
                        index   = evGroup[1][1]
                        ev      = evGroup[1][0]
                        rx      = rx1
                        cx      = cx1

                    self.validatedEigenpairs.append(index)

                    if store:
                        self.storedEigenpairs.append([ev, rx, cx])
                    else:
                        self.storedEigenpairs.append([ev])

    def _eigenpairNorm(self,i):
        """Reconstructs an eigenpair and checks what the l2 norm is. This is used when testing for ghost solutions.

        :param int i: which eigenpair to validate
        :return: the calculated l2 norm
        :rtype: float and the combined test vector
        """

        
        rx = PETSc.Vec().createSeq(self.dim[0])
        cx = PETSc.Vec().createSeq(self.dim[0])

        ev = self.eps.getEigenpair(i,rx,cx)

        real,imag = self._reconstructEigenvector(rx,cx)

        storeReal = real.copy() #We have to keep a backup - using restoreVector seems to free() the memory pointed to by these vectors
        storeImag = imag.copy()

        #We create the vector used to check whether the solution is a ghost or not
        testVec = PETSc.Vec().createSeq(self.dim[0])
        testVec.restoreSubVector(self.indicesReal,real)
        testVec.restoreSubVector(self.indicesImag,imag)

        #testVec.setType(PETSc.Vec.Type.SEQ) #change its type back to a normal seq
        resVec = testVec.duplicate() #allocate the emmory for the test vector

        self.AA.mult(testVec,resVec)

        norm = resVec.norm(PETSc.NormType.N2)


        return norm, storeReal, storeImag


    def solve(self,nEigen, store=True, maxSolveTime = 240):
        super().solve(nEigen, maxSolveTime)
        #We validate
        self.validateEigenpairs(store=store)

        return len(self.validatedEigenpairs)


    def getEigenpair(self,i,absSpectra=True):
        """Gets the i-th eigenpair.

        :param int i: index of the eigenpair to resturn
        :param bool absSpectra: (optional) whether to return the absolute spectra or spectra relative to shift
        """

        [ev, real, imag] = self.storedEigenpairs[i]

        if absSpectra:

            if self.mode == 'direct':
                ev += self.shift
            else:
                ev += self.shift.conjugate()

        return ev, dolf.PETScVector(real), dolf.PETScVector(imag)

class RealEigensolver(ComplexEigensolver):
    """This is an eigensolver that takes a real eigenvalue problem and enables a complex shift to be applied. This is implemented
    by extending the ComplexEigensolver and setting the imaginary part to zero.

    Because all the matrices are real, it morphs an NxN problem into a 2Nx2N problem. See :ref:`real-eigensolver-label` for
    details.

    :param dolfin.PETScMatrix A: The A operator
    :param dolfin.PETScMatrix B: (optional) The B operator of a generalised eigenvalue problem.
    :param str mode: (optional) whether to use direct mode (right eigenvectors) or adjoint mode (left eigenvectors)
    """

    def __init__(self,A,B=None,mode=None):
        super().__init__(AReal = A, AImag = 0, B = B, mode = mode)
