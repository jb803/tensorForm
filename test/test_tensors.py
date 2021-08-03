""" Tests all aspects of tensor manipulation"""
import os
import pytest

from tensorForm import tensor as t
from tensorForm.aux import RealEigensolver as Eigensolver
import numpy as np
import scipy.special

import dolfin as dolf


#In theory tests are in order from most basic to most complex.
#i.e. if an early test fails it implies bad behaviour in the later tests

def test_basicIndexManipulation():
    """Tests basic index manipulation
    1) Validation
    2) Availability of new index
    3) Addition compatability
    4) Multiplicative compatability
    """

    #1)Validation
    assert not t.validIndices(['a','b','?','D'])
    assert t.validIndices(['F','g','h','I'])
    assert not t.validIndices(['a','A'])

    #2) Availability
    assert t.getNextAvailableIndex(['a','b','D','c','E']) == 'f'
    assert t.getNextAvailableIndex(['b','E','c']) == 'a'
    with pytest.raises(t.IndexError):
        t.getNextAvailableIndex(['B','c','a','d','X','Q','r','s','g','h','e',
                                 'f','I','k','J','L','m','n','o','p','t','u',
                                 'v','w','y','z'])

    #3) Addition compatibility
    assert t.additionCompatibleIndices(['A','b','c','D'],['A','D','b','c'])
    assert not t.additionCompatibleIndices(['A','b','c','D'],['A','d','b','c'])
    assert not t.additionCompatibleIndices(['A','b','c','D'],['A','e','b','c'])

    with pytest.raises(t.IndexError):
        t.additionCompatibleIndices(['?'],['a'])
    with pytest.raises(t.IndexError):
        t.additionCompatibleIndices(['a'],['a','A'])

    #4) Multiplicative compatibility
    assert t.getMultiplicationContraction(['a','b'],['c','d','E']) == []
    assert t.getMultiplicationContraction(['a','B'],['c','b','A']) == [['a','A'],['B','b']]
    assert t.getMultiplicationContraction(['E','f'],['a','E']) == [['E','E']]

    with pytest.raises(t.IndexError):
        assert t.getMultiplicationContraction(['A','a'],['b'])

#We test the different tensor types
def test_constantTensor():
    #We test the constant tensor implementation, whether it evaluates and the comma derivatives are correct
    #1) Adding and basic derivatives
    one = t.ConstantTensor(1.,None,[])

    assert one({}) == 1.
    assert one['i']({'i':0})==0.

    assert one({},debug=True) == 'Re(1.0)'
    assert one({},debug=True,realPart=False) == 'Im(1.0)'

    #2) Failing to perform comma derivative
    with pytest.raises(t.TensorError):
        one['I']
    with pytest.raises(t.TensorError):
        one['?']

def test_additionTensor():
    """Test the addition of two tensorss
    #1) Adding of two constant tensors
    #2) Checking the derivative works
    #3) Checking failure to add two tensors due to index mismatch
    #4) Checking failure to add a tensor plus non tensor
    #5) Check negation
    #6) Check subtraction
    """
    one = t.ConstantTensor(1.,None)
    two = t.ConstantTensor(2.,None)

    three = one + two
    four = three + one

    #1) Addition of constant tensors
    assert three({},debug=True) == 'Re(' + str(1.) + '+' + str(2.) +')'
    assert three({}) == 3.
    assert four({}) == 4.

    #2) Checking derivative works
    threeDeriv = three['i']
    fourDeriv  = four['i']

    assert threeDeriv({'i':1}) == 0.
    assert fourDeriv({'i':1}) == 0.

    #3) Checking failure to add two tensors due to index mismatch
    fourDeriv2 = four['j']
    with pytest.raises(t.TensorError):
        four + fourDeriv2

    #4) Checking failure to add a tensor to a non tensor
    with pytest.raises(t.TensorError):
        1 + one 

    #5) Negation
    minusOne = -one
    assert minusOne({}) == -1.

    #6) Subtraction
    fourMinusThree = four - three
    assert fourMinusThree({}) == 1.

def test_productTensor():
    """We test the ProductTensor. We
    #1) Test multiplication of some constants
    #2) Check index change is correct
    """
    two   = t.ConstantTensor(2.,None,['i'])
    three = t.ConstantTensor(3.,None,['j'])

    six = two*three

    #1)Test multiplication of some constants
    assert six({'i':0,'j':0}) == 6.

    eighteen = 3*six
    assert eighteen({'i':0,'j':0}) == 18.

    #2) Test index change is correct
    assert eighteen.indices == ['i','j']

    assert eighteen['z'].indices == ['i','j','z']


    #)3) Test division of two scalars
    two   = t.ConstantTensor(2.,None,[])
    three = t.ConstantTensor(3.,None,[])
    dEval = three/two
    assert dEval({}) == 1.5

def test_remapTensor():
    """We test the RemapTensor. TODO: introduce more tests when we have more capable tensors
    than just the ConstantTensor.

    We first of all test whether we get errors when we try and do a blatantly wrong remapping. Like
    remapping between non existent indices.

    1)Remapping non existant indices
    2)Checking a simple remap
    3)Checking a required remap due to comma derivative
    """

    def indexReporting(indexAssignment,debug=False):
        """This is a helper function for testing remapping.

        When evaluation to check the index matching process, rawEvaluation 
        should be set to True"""
        return indexAssignment

    
    #The tensor we'll be remapping
    one = t.ConstantTensor(1.0,None,['a','B'])
    one.evaluate = indexReporting

    #1) Non existant or invalid indices
    with pytest.raises(t.TensorError):
        remapping = t.RemapTensor(one,'?','a',None)
    with pytest.raises(t.TensorError):
        remapping = t.RemapTensor(one,'c','a',None)
    with pytest.raises(t.TensorError):
        remapping = t.RemapTensor(one,'a','b',None)
    with pytest.raises(t.TensorError):
        remapping = t.RemapTensor(one,'a','C',None)

    #2) we perform a simple remap
    remapped = t.RemapTensor(one,'B','C',None)
    #We perform some tests
    assert remapped.indices == ['a','C']
    assert remapped({'a':0.5, 'c':1.0}, rawEvaluation = True) == {'a':0.5,'b':1.0}

    remapped = t.RemapTensor(remapped,'a','b',None)
    assert remapped.indices == ['b','C']
    assert remapped({'b':0.2,'c':0.9}, rawEvaluation = True) == {'a':0.2,'b':0.9}

    #3) We apply a derivative
    deriv = remapped['a']

    assert deriv.indices       == ['b','C','a']
    assert deriv.t.indices     == ['b','C','d']
    assert deriv.t.t.indices   == ['a','C','d']
    assert deriv.t.t.t.indices == ['a','B','d']

    deriv.t.t.t.evaluate = indexReporting
    assert deriv({'b' : 1.0, 'c': 2.0,'a':-3.}, rawEvaluation=True) == {'a':1.0,'b':2.0,'d':-3.}

def test_MatrixTensor():
    """We test the matrix tensor.
    1) Matrix matches index dimensions
    2) Matrix matches coordinate system dimensions
    3) Evaluation
    4) Correct addition
    5) Correct debug string
    """
    cartesian = t.Cartesian()

    badVec   = np.array([0.,1,3.]) #Too long, bigger than the number of spatial dimensions
    badMat   = np.array([[0.,1.],[2.,3.],[4.,5.]]) #as above
    goodVec  = np.array([0.,1.])
    goodMat1 = np.array([[0., 1.],[2., 3.]])
    goodMat2 = np.array([[-1.,2.],[4., 0.]])

    #1) Failure if matrix doesn't match index dimensions
    with pytest.raises(t.TensorError):
        t.MatrixTensor('A',goodVec,['a','b'],cartesian)
    with pytest.raises(t.TensorError):
        t.MatrixTensor('A',goodMat1,['a'],cartesian)

    #2) Failure if matrix does not match spatial dimensions
    with pytest.raises(t.TensorError):
        t.MatrixTensor('A',badVec,['a'],cartesian)
    with pytest.raises(t.TensorError):
        t.MatrixTensor('A',badMat,['a','b'],cartesian)

    #3) evaluation
    mat1 = t.MatrixTensor('A',goodMat1,['a','b'],cartesian)
    print(mat1({'b':0,'a':1}))
    assert mat1({'b':0,'a':1}) == 2.
    assert mat1({'a':0,'b':0}) == 0.

    mat = t.MatrixTensor('A',goodMat1,['a','B'],cartesian)
    assert mat({'b':0,'a':1}) == 2.

    #4)Addition check
    mat2 = t.MatrixTensor('B',goodMat2,['b','a'],cartesian) #NOTE: we've swapped the coordiantes here

    matrixSum = mat1 + mat2

    assert matrixSum({'a':0,'b':1}) == 5.#This should be mat1[0,1]+mat2[1,0] = 5

    #5)Debug string
    assert mat2({'a':0,'b':1},debug=True) == 'Re(B_b1_a0)'


def test_ContractionTensor():
    """Tests simple cases with the contraction tensor and some matrices

    #1) Error if we try and contract two many indices or indices that aren't valid/present
    #2) Contraction of a single tensor, getting the trace
    #3) Contracting of a product, getting the dot product
    """

    cartesian = t.Cartesian()

    mat  = np.array([[0.,1.],[-5.,2.]])
    vec1 = np.array([5.,2.])
    vec2 = np.array([6.,-2.])

    matTensor = t.MatrixTensor('A',mat,['i','J'],cartesian)
    vec1Tensor= t.MatrixTensor('b',vec1,['i'],cartesian)
    vec2Tensor= t.MatrixTensor('c',vec2,['J'],cartesian)

    #1) Trying to contract too many indices or invalid indices
    with pytest.raises(t.TensorError):
        t.ContractionTensor(matTensor,['i'],cartesian) #too few indices
    with pytest.raises(t.TensorError):
        t.ContractionTensor(matTensor,['i','J','k'],cartesian) #too many indices
    with pytest.raises(t.TensorError):
        t.ContractionTensor(matTensor,['i','j'],cartesian) #Not correct co vs contra

    #2) Contraction of a single tensor to get the trace
    trace = t.ContractionTensor(matTensor,['i','J'],cartesian)

    assert trace.indices == []
    assert trace({}) == 2.
    assert trace({},debug=True) == 'Re(A_i0_J0+A_i1_J1)'

    #3) dot product
    preDot = vec1Tensor*vec2Tensor
    dot = t.ContractionTensor(preDot,['i','J'],cartesian)

    assert dot.indices == []
    assert dot({}) == 26.
    assert dot({},debug=True) == 'Re(b_i0*c_J0+b_i1*c_J1)'


def test_tensorAlgebra():
    """Tests tensor algebra with overloaded operators"""
    cartesian = t.Cartesian()
    vec1 = np.array([5.,2.])
    vec2 = np.array([6.,-2.])
    a= t.MatrixTensor('a',vec1,['i'],cartesian)
    b= t.MatrixTensor('b',vec2,['i'],cartesian)

    mat1 = np.array([[0., 1.],[2., 3.]])
    mat2 = np.array([[-1.,2.],[4., 0.]])

    A = t.MatrixTensor('A',mat1,['i','J'],cartesian)
    B = t.MatrixTensor('B',mat2,['I','j'],cartesian)



    dot = a*b
    assert dot.indices==[]
    assert dot({}) == 26.

    doubleDot = A*B
    assert doubleDot.indices == []
    assert doubleDot({}) == 10.

def test_differentiableMatrixTensor():
    """This tests the implementation of the DifferentiableMatrixTensor.
    We have to construct a FEniCs object for this. We make a simple function space
    on a crude mesh.

    We first of all test the correct errors are raised on inappropriate construction
    1) Invalid indices, in particular upper case derivative indices
    2) An implied contraction on construction
    3) A scalar but with incorrect indices
    4) A badly shaped matrix
    5) A few general examples that shouldn't raise errors on contstruction

    We then construct a coordinate system, mesh, and simple funtion space.

    TODO: to speed up testing the mesh and function spaces could be implemented as pytest fixtures
    """
    cartesian   = t.Cartesian()

    cScalar     = np.array([1])
    mat         = np.eye(2)
    badShape1   = np.array([0,1,2]) #Exceeds coordinate system dimensions
    badShape2   = np.array([1,2]) 

    #1) Upper case derivative indices should cause a TensorError
    with pytest.raises(t.TensorError):
        t.DifferentiableMatrixTensor('A',mat,['i','j'],['C'],cartesian)
    #2) An implied contraction on construction should also cause a TensorError
    with pytest.raises(t.TensorError):
        t.DifferentiableMatrixTensor('A',mat,['i','j'],['i'],cartesian)
    #3) A scalar but with indices given as the array indices should cause an errro
    with pytest.raises(t.TensorError):
        t.DifferentiableMatrixTensor('A',cScalar,['i'],[],cartesian)
    #4) A badly shaped matrix should raise an error like MatrixTensor
    with pytest.raises(t.TensorError):
        t.DifferentiableMatrixTensor('A',badShape1,['i'],[],cartesian)
    with pytest.raises(t.TensorError):
        t.DifferentiableMatrixTensor('A',badShape2,['i','j'],[],cartesian)

    #5) A few things that shouldn't raise errors on construction and checking to see if they get given the correct signature
    dTens = t.DifferentiableMatrixTensor('A',mat,['i','j'],['a','b'],cartesian)
    assert dTens.indices == ['i','j','a','b']
    assert dTens({'i':0,'j':1,'a':2,'b':3},debug=True) == 'Re(d2A_i0_j1_Da2_Db3)'

    dTensComDeriv = dTens['c']
    assert dTensComDeriv.indices == ['i','j','a','b','c']

    dTensCovDeriv = dTens**'c'
    assert dTensCovDeriv.indices == ['i','j','a','b','c']

    #6) We now try this all with Functions from FEniCS
    #We should be able to perform a covariant derivative and evaluate without an error

    mesh    = dolf.UnitSquareMesh(5,5)
    fs      = dolf.FunctionSpace(mesh,'CG',1)
    f       = dolf.Function(fs)

    #We test a scalar tensor (this is the most likely to break)
    fMat = np.array([f])
    fenTensor = t.DifferentiableMatrixTensor('u',fMat,[],[],cartesian)

    deriv = fenTensor**'a'
    deriv({'a':1}, rawEvaluation=True) #These aren't set up via complex quantities yet so rawEvaluation is required

    #7) we now test a 'vector' tensor
    fVec = np.array([f,f])
    vecTensor = t.DifferentiableMatrixTensor('u',fVec,['i'],[],cartesian)
    deriv = vecTensor**'a'

    deriv({'i':1,'a':0})


def test_covariantDerivative():
    """We attempt to do some basic tests on the coviant derivative.
    #1) Errors with derivative indices
    2) Correct indices for a non contracting covariant derivative
    3) Correct indices for a contracted derivative
    4) Correct debug evaluations
    """

    cartesian = t.Cartesian()
    npVec = np.array([0,1])

    tens = t.DifferentiableMatrixTensor('A',npVec,['i'],[],cartesian)

    #1) Error if attempting to perform a cov derivative with an upper case index
    with pytest.raises(t.TensorError):
        tens**'A'

    #2) Correct indices for a non contracting covariant dervative
    nonContracting = tens**'j'
    assert nonContracting.indices == ['i','j']

    #3) Correct indices for a contracting covariant derivative
    contracting = tens**'i'
    assert contracting.indices == []

    #4) Correct debug evaluations
    #TODO: manually step through the tree to work these strings out!
    #assert nonContracting({'i':0,'j':1},debug=True) == 'asdas'
    #assert contracting({},debug=True) == 'asd'        

def test_diffusionProblem():
    """Tests a basic diffusion problem, to make sure everything works 
    correctly.

    Solves scalar diffusion in cartesian coordinates for a simple mesh
    result should be a linear variation of concentration

    Solves scalar diffusin in cylindrical coordiantes for a simple mesh
    result should be....
    """

    #Set up mesh and function space
    mesh = dolf.UnitSquareMesh(50,50)
    fs   = dolf.FunctionSpace(mesh,'CG',1)

    U = dolf.TrialFunction(fs)
    V = dolf.TestFunction(fs)
    soln = dolf.Function(fs)
    soln.rename('u','u')

    def botSurf(x,on_boundary):
        return dolf.near(x[1],0.) and on_boundary

    def topSurf(x,on_boundary):
        return dolf.near(x[1],1.) and on_boundary

    rhs = V*dolf.Constant(0.)*dolf.dx

    bcs = [dolf.DirichletBC(fs,dolf.Constant(1.),botSurf),
           dolf.DirichletBC(fs,dolf.Constant(0.),topSurf)]


    #1) Cartesian
    cartesian = t.Cartesian()

    u = t.DifferentiableMatrixTensor('u',np.array([U]),[],[],cartesian)
    v = t.DifferentiableMatrixTensor('v',np.array([V]),[],[],cartesian)

    uj = u**'j'
    vi = v**'i'

    met = cartesian.MetricTensor(['I','J'])

    form = uj*met*vi

    lhs = form({})*dolf.sqrt(cartesian.MetricDeterminant()({}))*dolf.dx

    dolf.solve(lhs==rhs,soln,bcs)

    actSoln = dolf.Expression('1. - x[1]',degree=1) #The actual solution

    #We can compute the errors using tensors as well
    solnTensor     = t.DifferentiableMatrixTensor('u0',np.array([soln]),[],[],cartesian)
    actSolnTensor  = t.DifferentiableMatrixTensor('ud',np.array([actSoln]),[],[],cartesian)

    err = dolf.assemble( ((solnTensor-actSolnTensor)*(solnTensor-actSolnTensor))({}, rawEvaluation = True)*dolf.sqrt(cartesian.MetricDeterminant()({}))*dolf.dx)
    
    assert np.sqrt(err) < 1e-12

    #2) Cylindrical
    r = dolf.SpatialCoordinate(mesh)[1] + dolf.Constant(0.5)

    cylindrical = t.Cylindrical(r)

    u = t.DifferentiableMatrixTensor('u',np.array([U]),[],[],cylindrical)
    v = t.DifferentiableMatrixTensor('v',np.array([V]),[],[],cylindrical)

    uj = u**'j'
    vi = v**'i'

    met = cylindrical.MetricTensor(['I','J'])
    form = uj*met*vi

    lhs = form({})*dolf.sqrt(cylindrical.MetricDeterminant()({}))*dolf.dx #As this isn't set up as a proper complex quantity these are required

    dolf.solve(lhs==rhs,soln,bcs)

    #We can compute the errors using tensors as well
    actSoln = dolf.Expression('-0.91*log10(x[1]+0.5)/log10(2.71828) + 0.369',degree=1) #The actual solution

    solnTensor     = t.DifferentiableMatrixTensor('u0',np.array([soln]),[],[],cylindrical)
    actSolnTensor  = t.DifferentiableMatrixTensor('ud',np.array([actSoln]),[],[],cylindrical)

    err = dolf.assemble( ((solnTensor-actSolnTensor)*(solnTensor-actSolnTensor))({}, rawEvaluation=True)*dolf.sqrt(cylindrical.MetricDeterminant()({}))*dolf.dx)
    print(np.sqrt(err))

    #TODO: we can make this test more precise by replacing A and B in the known expression with
    #more accurate values
    assert np.sqrt(err) < 1e-4

def test_kroneckerDelta():
    """Tests the kronecker delta implementation.
    #1) Make sure we get an error if we do not have form (p,p) indices
    #2) Raise a not implemented error if p> 1
    #3) Test a contraction
    #4) Test comma derivative raises an error
    """
    cartesian = t.Cartesian()

    #1 )A series of bad indicise:
    #1a)Implied contraction
    with pytest.raises(t.TensorError):
        t.KroneckerDelta(['I','i'],cartesian)
    #1b)Indices not of form (p,p)
    with pytest.raises(t.TensorError):
        t.KroneckerDelta(['I','J','k'],cartesian)
    #1c)Zero indices
    with pytest.raises(t.TensorError):
        t.KroneckerDelta([],cartesian)

    #2)Not implemented error for a generalised KronceckerDelta
    with pytest.raises(NotImplementedError):
        t.KroneckerDelta(['I','K','j','l'],cartesian)

    #3) We test a simple contraction
    delta = t.KroneckerDelta(['I','j'],cartesian)
    cont  = t.ContractionTensor(delta,['I','j'],cartesian)

    assert cont.indices == []
    assert cont({},debug=True) == 'Re(delta_I0_j0+delta_I1_j1)'
    assert cont({}) == 2.

    #4) Test comma derivative raises an error
    with pytest.raises(t.TensorError):
        delta['q']

def test_complexQuantity():
    """Tests the complex quantity implementation.
    #1) Perform various complex math using the built in type and the implemented type
    #2) Test multiplying between complex type and ComplexQuantity
    """

    #1)
    a_py = 1  + 2j
    b_py = -3 + 6j
    c_py = 1j
    d_py = 2

    a_t  = t.ComplexQuantity(1 ,2)
    b_t  = t.ComplexQuantity(-3,6)
    c_t  = t.ComplexQuantity(0,1)
    d_t  = t.ComplexQuantity(2,0)

    assert (a_t+b_t).real == (a_py+b_py).real 
    assert ((a_t-c_t)*d_t).real == ((a_py-c_py)*d_py).real
    assert (b_t*d_t-a_t+c_t).imag == (b_py*d_py - a_py + c_py).imag

    #2)
    assert (a_t*b_py).real == (a_py*b_py).real
    assert (a_t*c_py).imag == (a_py*c_py).imag 
    assert (a_t*d_py).real == (a_py*d_py).real


def test_permutationSymbol():
    """Tests the implementation of the permutation symbol.

    #1) Make sure the dimensions are correct for a 2D and 3D permutation
    #2) Make sure we throw an error if the number of indices is wrong
    #3) Assert it produces the correct values of different evaluations

    """
    cart2 = t.Cartesian(dim=2)
    cart3 = t.Cartesian(dim=3)

    #1)
    p2 = t.PermutationSymbol(['I','J'],cart2)
    p3 = t.PermutationSymbol(['i','j','k'],cart3)

    assert p2.objMat.shape == (2,2)
    assert p3.objMat.shape == (3,3,3)

    #2) 
    with pytest.raises(t.TensorError):
        t.PermutationSymbol(['I'],cart2)
    with pytest.raises(t.TensorError):
        t.PermutationSymbol(['I','J','K','L'],cart3)

    #3)
    #a - 2d tensor - we check all elements
    assert p2({'i':0,'j':0}) == 0
    assert p2({'i':0,'j':1}) == 1
    assert p2({'i':1,'j':0}) ==-1
    assert p2({'i':1,'j':1}) == 0

    #b - 3d tensor - we check some elements
    assert p3({'i':0,'j':0,'k':0}) == 0
    assert p3({'i':1,'j':1,'k':0}) == 0
    assert p3({'i':0,'j':1,'k':0}) == 0

    assert p3({'i':0,'j':1,'k':2}) == 1
    assert p3({'i':1,'j':2,'k':0}) == 1
 
    assert p3({'i':1,'j':0,'k':2}) == -1
    assert p3({'i':0,'j':2,'k':1}) == -1

def test_leviCivita():
    """Tests construction of the Levi Civita tensor.

    #1) Make sure the tensor throws an error with mixed symbols
    #2) Make sure we can build it for cartesian coordinates
    """
    cart2 = t.Cartesian(dim=2)
    cart3 = t.Cartesian(dim=3)

    #1)
    with pytest.raises(t.TensorError):
        cart2.LeviCivita(['i','J'])
    with pytest.raises(t.TensorError):
        cart3.LeviCivita(['I','J','k'])

    #2)
    cart2.LeviCivita(['i','j'])
    cart2.LeviCivita(['I','J'])
    cart3.LeviCivita(['i','j','m'])

@pytest.mark.slow
def test_wavenumbers(m=1):
    """Tests a wavenumber problem in cylindrical coordinates by doing an eigenvalue problem.

    The solution to this is lambda = - (npi)^2 - Bz^2, where Bz is a zero of the first Bessel function of order m where
    m is the azimuthal wavenumber"""
    mesh = dolf.UnitSquareMesh(40,40)
    fs   = dolf.FunctionSpace(mesh,'CG',2)

    r = dolf.SpatialCoordinate(mesh)[1] 
    cylindrical = t.Cylindrical(r, disableThetaDeriv=False)

    U = dolf.TrialFunction(fs)
    V = dolf.TestFunction(fs)

    

    def botSurf(x, on_boundary):
        return dolf.near(x[1], 0.) and on_boundary
    def topSurf(x, on_boundary):
        return dolf.near(x[1], 1.) and on_boundary
    def leftSurf(x, on_boundary):
        return dolf.near(x[0], 0) and on_boundary
    def rightSurf(x, on_boundary):
        return dolf.near(x[0], 1) and on_boundary

    bcs = [dolf.DirichletBC(fs,dolf.Constant(0.),surf) for surf in [botSurf, topSurf, leftSurf, rightSurf]]

    azimuthalWavenumber = t.ComplexQuantity(0, m)

    wavenumbers = [None, azimuthalWavenumber, None]

    u = t.DifferentiableMatrixTensor('u',np.array([t.ComplexQuantity(U, 0, cSystem = cylindrical, customWavenumbers = wavenumbers)]),[],[],cylindrical)
    v = t.DifferentiableMatrixTensor('v',np.array([t.ComplexQuantity(V, 0, cSystem = cylindrical, customWavenumbers = wavenumbers, negativeWavenumber = True)]),[],[],cylindrical)

    BForm = (u*v)({})*dolf.sqrt(cylindrical.MetricDeterminant()({}))*dolf.dx
    AForm = (-u**'i' * v**'j' * cylindrical.MetricTensor(['I','J']))({})*dolf.sqrt(cylindrical.MetricDeterminant()({}))*dolf.dx

    AA = dolf.PETScMatrix()
    BB = dolf.PETScMatrix()

    dolf.assemble(AForm, tensor=AA)
    dolf.assemble(BForm, tensor=BB)

    for bc in bcs:
        bc.apply(AA)
        bc.zero(BB)

    eigensolver = Eigensolver(AA,BB)
    nEv = eigensolver.solve(10)

    eigenvalues = []
    for i in range(nEv):
        ev, rx, cx = eigensolver.getEigenpair(i)
        eigenvalues.append(ev.real)

    #We now compute the first analytical eigenvalues
    analyticEV = []
    besselZeros = scipy.special.jn_zeros(m,8)
    for i in range(1,8):
        for j in range(8):
            analyticEV.append(-((i*np.pi)**2. + besselZeros[j]**2.))

    eigenvalues.sort(key = lambda x: -x)
    analyticEV.sort(key = lambda x: -x)

    for i in range(len(eigenvalues)):
        assert np.abs(analyticEV[i]-eigenvalues[i]) < 1e-2



