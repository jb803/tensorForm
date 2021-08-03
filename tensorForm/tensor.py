"""
Manages the tensor calculus framework. 

We have to be very careful here, any bugs in this code could be a real pig to hunt down.

"""

#TODO: when the contraction tensor is done we need to sort out contraction of comma derivative!!!
#TODO: check whether copy.deepcopy is always necessary. May be overcautious

import copy
from abc import ABC, abstractmethod

import numpy as np
import dolfin as dolf
from sympy.combinatorics.permutations import Permutation

import ufl 
import sys

import time

Permutation.print_cyclic = False

_alphabet = list('abcdefghijklmnopqrstuvwxyz')

##############Errors:

class IndexError(Exception):
    """This error is used if the error arises due to mismatching or invalid indices"""
    def __init__(self,message):
        self.message = message

class TensorError(Exception):
    """This error is used for all other errors in the module"""
    def __init__(self,message):
        self.message = message

################Functions for index manipulation

def validIndices(indices):
    """Tests whether the list contains single alphabetic characters that are either upper or lower case
    Args:
        indices: the index list to validate
    """
    for a in indices:
        if a.lower() not in _alphabet:
            return False
        if indices.count(a.lower()) + indices.count(a.upper()) != 1:
            return False
    return True

def getNextAvailableIndex(indices):
    """Gets the next available index that's not been used so far.
    Args:
        indices: the indices used so far
    """
    
    for a in _alphabet:
        found = False
        for b in indices:
            if b.lower() == a:
                found = True
        if not found:
            return a
    #If we get here there are no available indices
    raise IndexError('No more available indices')

def additionCompatibleIndices(indices1,indices2):
    """We check weheter the indices are compatible to add. For indices to be comaptible,
    the order can be different but the case must be the same"""
    #We check both indices are valid
    if not validIndices(indices1) or not validIndices(indices2):
        raise IndexError('Invalid indices. Cannot compare')

    for i in indices1:
        if indices2.count(i) != 1:
            return False
    for i in indices2:
        if indices1.count(i) != 1:
            return False
    return True

def getMultiplicationContraction(indices1,indices2):
    """We check whether the indices are comaptible for multiplication. For inidices to be compatible,
    they both just have to be valid. The restrictions are therefore much less than those of addition.
    This just returns the indices that we're contracting due to the multiplication
    Returns:
        Contraction index pairs
    """
    if not validIndices(indices1) or not validIndices(indices2):
        raise IndexError('Invalid indices. Cannot compare')

    contractionPairs = []

    for i in indices1:
        if i.lower() in indices2 or i.upper() in indices2:
            if i.lower() in indices2:
                contractionPairs.append([i,i.lower()])
            else:
                contractionPairs.append([i,i.upper()])
    return contractionPairs

def isInIndices(index,indices):
    if not validIndices(indices) or not validIndices([index]):
        raise IndexError('Invalid indices. Cannot compare')
    if indices.count(index.lower()) + indices.count(index.upper()) >= 1:
        return True
    return False


############Base tensor class
class Tensor(ABC):
    """The class defining the behaviour of a tensor"""
    def __init__(self,name,indices,cSystem):
        if not validIndices(indices):
            raise IndexError('Cannot create tensor, indices are invalid')
        self.indices = indices
        self.cSystem = cSystem
        self.name = name

    @abstractmethod
    def evaluate(self,indexAssignment,debug=False):
        #Evaluates the tensor, if debug is on we are expected to return a text string
        pass

    @abstractmethod
    def commaDerivative(self,index):
        #Performs comma derivative 
        pass

    def _validateNewIndex(self,indexToAdd):
        """Validates adding a new index, a helper function for this class
        :param str indexToAdd: the new index we want to add
        """
        for i in self.indices:
            if indexToAdd == i.lower():
                raise TensorError('Cannot add this index: ' + indexToAdd)

    def __call__(self,indexAssignment,debug = False, realPart = True, rawEvaluation = False):
        """This is the method that should be used to evaluate the tensor. Evaluate shouldn't be called directly
        as this is where all the important checks take place i.e. we're not being passed extra indices
        that might cause failures later down the line.

        Only the item at the top of the evluation tree evaluated via __call__. Everything else should use the
        internal evaluate function

        :param dict(float) indexAssignment: for tensors with free indices, this is the value associated with each index
        :param bool debug: (optional) whether to return the debug evluation (normally a str)
        :param bool realPart: (optional) whether to return the real or imaginary part. default is real
        :param bool rawEvaluation: (optional) whether to return the raw evaluation (i.e. without extracting real or imaginary). 

        """
        evalStart = time.time()
        for index in indexAssignment.keys():
            if not isInIndices(index,self.indices):
                raise TensorError('Extra keys are being passed during evaluation!!')
        for index in self.indices:
            if not isInIndices(index,list(indexAssignment.keys())):
                raise TensorError('Missing index, cannot evaluate')

        evaluation = self.evaluate(indexAssignment, debug=debug)

        if rawEvaluation:
            retVal = evaluation
        else:

            if debug:
                if realPart: prePend = 'Re('
                else:        prePend = 'Im('

                retVal = prePend + evaluation +')'

            else:
                if realPart: retVal = evaluation.real
                else:        retVal = evaluation.imag 


        return retVal


    def __add__(self,other):
        """Perform addition"""
        if not isinstance(other,Tensor):
            raise TensorError('Unable to add these two objects together')
        return AdditionTensor(self,other,self.cSystem)

    def __radd__(self,other):
        return self + other

    def __neg__(self):
        """Implements negation"""
        neg1Tensor = ConstantTensor(-1.,self.cSystem)

        return self*neg1Tensor

    def __truediv__(self, other):
        """True division. Only works it this is a scalar tensor and the other is a scalar tensor"""
        if not isinstance(other,Tensor):
            raise TensorError('Can only do division if both objects are tensors')

        if self.indices != [] or other.indices != []:
            raise TensorError('Can only do division if both objects are scalar tensors')

        #We're safe
        divFunc = lambda x : 1./x

        return self*ScalarFunctionTensor('1./' + self.name, other, divFunc, self.cSystem)




    def __sub__(self,other):
        negOther = -other
        return self + negOther

    def __mul__(self,other):
        """Perform multiplication"""
        if type(other) == float or type(other) == int or type(other) == complex:
            return self * ConstantTensor(other,self.cSystem)
        elif isinstance(other,Tensor):
            #We now sort out stuff
            contractions = getMultiplicationContraction(self.indices,other.indices)
            if contractions == []:

                return ProductTensor(self,other,self.cSystem)
            else:
                #We have to work out what indices are used. We always perfom remaps only on the second object
                usedIndices  = copy.deepcopy(self.indices)
                ob2IndexList = copy.deepcopy(other.indices)

                for c in contractions:
                    ob2IndexList.remove(c[1])


                usedIndices.extend(ob2IndexList)

                #We have the used indices
                remapIndices = []
                remap = other
                for c in contractions:
                    newIndex = getNextAvailableIndex(usedIndices)
                    if c[1].isupper():
                        newIndex = newIndex.upper()
                    else:
                        newIndex = newIndex.lower()

                    remapIndices.append(newIndex)
                    usedIndices.append(newIndex)

                    remap = RemapTensor(remap,c[1],newIndex,self.cSystem)

                #We perform the remaps
                
                product = ProductTensor(self,remap,self.cSystem)

                contraction = product

                for i in range(len(remapIndices)):
                    c = [contractions[i][0],remapIndices[i]]
                    contraction = ContractionTensor(contraction,c,self.cSystem)

                return contraction            
        else:
            raise TensorError('Unable to multiply these objects together')
    
    def __rmul__(self,other):
        return self*other

    def __getitem__(self,index):
        """This is how the comma derivative is applied. Here we check whether this is going to invoke a
        contraction. (In which case we call the DeriviativeContraction tensor object). Otherwise, we
        can call the implemented comma derivative"""
        if not index.islower():
            raise TensorError('Comma derivative can only be applied with a lowercase index ' + index)
        if index not in _alphabet:
            raise TensorError('Cannot apply comma derivative with index '  + index )
        if isInIndices(index,self.indices):
            copyIndices = copy.deepcopy(self.indices)
            copyIndices.append(index)
            nextAvailableIndex = getNextAvailableIndex(copyIndices).lower()

            deriv = self[nextAvailableIndex]

            #We have to ensure we refer to component correclty. This is a subtle point
            #If we start to swap around covariant and contravariant indices then we
            #may break the covariant derivative!
            if index.lower() in self.indices:
                index = index.upper()
            else:
                index = index.lower()
            return ContractionTensor(deriv,[index,nextAvailableIndex],self.cSystem)


        return self.commaDerivative(index)


    def __pow__(self,power):
        """This is the covariant derivative"""
        if not power.islower() or power.lower() not in _alphabet:
            raise TensorError('Cannot apply covariant derivative with ' + power)

        #We check if there's an implied contraction
        if isInIndices(power,self.indices):
            #We have to do a contraction!
            #We remap power
            copyIndices = copy.deepcopy(self.indices)
            copyIndices.append(power)

            newIndex = getNextAvailableIndex(copyIndices)

            covDeriv = self**newIndex

            #newIndex is the covariant derivative index, power needs to have the same case as the
            #version in the index list
            if power.lower() in self.indices:
                power = power.lower()
            else:
                power = power.upper()

            return ContractionTensor(covDeriv,[power,newIndex],self.cSystem)

        #We have to apply the covariant derivative
        covDeriv = self[power] #We get the comma derivative

        #We get what the swapping index will be:
        copyIndices = copy.deepcopy(self.indices)
        copyIndices.append(power) #We don't want to acidentally invoke a contraction with the powers

        swapIndex = getNextAvailableIndex(copyIndices)


        for index in self.indices:
            if index.isupper():
                #We're doing the covariant derivative of a contravariant index
                christoffel = self.cSystem.ChristoffelSymbol([index,power,swapIndex.lower()])
                remap = RemapTensor(self,index,swapIndex.upper(),self.cSystem) #the remap is required because we only contract over the swap index

                covDeriv = covDeriv + christoffel*remap
            elif index.islower():
                #We're ding the covariant derivative of a covariant index
                christoffel = self.cSystem.ChristoffelSymbol([swapIndex.upper(),power,index])
                remap = RemapTensor(self,index,swapIndex.lower(),self.cSystem)

                covDeriv = covDeriv - christoffel*remap
            else:
                raise TensorError('Corruption ocurred in index list, ' + index + ' is not a valid index')
        return covDeriv

####### Mechanical Tensors ###############
#These subclasses of Tensor perform the grunt work to get the tensor calculus mechanics to work

class RemapTensor(Tensor):
    """This tensor remaps a single index"""
    def __init__(self,tensor,oldIndex,newIndex,cSystem):
        #We validate the remap
        #oldIndex us the index in the tensor object
        #newIndex is the index we replace it with in our signature
        if oldIndex.lower() not in _alphabet or newIndex.lower() not in _alphabet:
            raise TensorError('Cannot map from ' + oldIndex + ' to ' + newIndex)
        if oldIndex not  in tensor.indices:
            raise TensorError('Swap does not make sense, ' + oldIndex + ' is not present')
        if isInIndices(newIndex,tensor.indices):
            raise TensorError('Cannot perform remapping, new index is already present')
        if (oldIndex.islower() and not newIndex.islower()) or (oldIndex.isupper() and not newIndex.isupper()):
             raise TensorError('Indices must be same type i.e. both covariant or both contravariant')

        name = 'REMAP(' + oldIndex + ',' + newIndex + ')_' + tensor.name
        indexList = copy.deepcopy(tensor.indices)

        pos = indexList.index(oldIndex)
        indexList[pos] = newIndex

        self.t = tensor
        self.newIndex = newIndex
        self.oldIndex = oldIndex

        super().__init__(name,indexList,cSystem)

    def evaluate(self,indexAssignment,debug=False):
        """We remap the old inex to the new index"""
        passedOnIndex = {}

        for a in indexAssignment.keys():
            if a == self.newIndex.lower():
                passedOnIndex[self.oldIndex.lower()] = indexAssignment[a]
            else:
                passedOnIndex[a] = indexAssignment[a]

        retVal =  self.t.evaluate(passedOnIndex,debug=debug)

        if debug:
            retVal = 'REMAP['+self.newIndex+'>'+self.oldIndex+']('+retVal +')'

        return retVal

    def commaDerivative(self,index):
        #We know this cannot be a contraction, as that is handled by the Tensor base class
        #We just need to check the new index isn't the oldIndex
        if index == self.oldIndex.lower():
            #We're going to have to do another remapping
            #We get the next available index
            copyIndices = copy.deepcopy(self.indices)
            copyIndices.append(self.oldIndex.lower())
            newIndex = getNextAvailableIndex(copyIndices)

            deriv = RemapTensor(self.t[newIndex],self.oldIndex,self.newIndex,self.cSystem)

            return RemapTensor(deriv,newIndex,index,self.cSystem)

        else:
            #We just have to take the derivative of the tensor object
            return RemapTensor(self.t[index],self.oldIndex,self.newIndex,self.cSystem)

#The addition tensor
class AdditionTensor(Tensor):
    """This represents the addition of two tensors"""
    def __init__(self,t1,t2,cSystem):
        #We assert that we can add these two indices are compatible with addition
        if not additionCompatibleIndices(t1.indices,t2.indices):
            raise TensorError('Unable to add tensors, indices not compatible')

        self.t1 = t1
        self.t2 = t2

        name = t1.name+'+'+t2.name
        indexList = copy.deepcopy(t1.indices)

        super().__init__(name,indexList,cSystem)

    def evaluate(self,indexAssignment,debug=False):
        """We evaluate the sum of the two tensors. If debug 
        is enabled then we  construct the string representing the sum
        """
        t1Eval = self.t1.evaluate(indexAssignment,debug=debug)
        t2Eval = self.t2.evaluate(indexAssignment,debug=debug)

        if debug:
            return t1Eval +'+'+t2Eval
        else: 
            return t1Eval + t2Eval

    def commaDerivative(self,index):
        """We perform the comma derivative on both"""
        return self.t1.commaDerivative(index) + self.t2.commaDerivative(index)

#The product tensor
class ProductTensor(Tensor):
    """The product tensor represents the product of two tensors where no contraction is taking place"""
    def __init__(self,t1,t2,cSystem):
        if getMultiplicationContraction(t1.indices,t2.indices) != []:
            raise TensorError('ContractionTensor should be used when multiplying and contracting!')
        self.t1 = t1
        self.t2 = t2

        name = t1.name +'*' + t2.name
        indexList = copy.deepcopy(t1.indices)

        #For some reason you can't do .extend([])
        for a in copy.deepcopy(t2.indices):
            indexList.append(a)

        super().__init__(name,indexList,cSystem)

    def evaluate(self,indexAssignment,debug=False):
        """We evaluate the sum of the two tensors. If debug 
        is enabled then we  construct the string representing the sum.
        Some index manipulation is done in order to avoid passing through extra keys.
        """

        t1Indices = {}
        t2Indices = {}

        for index in indexAssignment.keys():
            if isInIndices(index,self.t1.indices):
                t1Indices[index] = indexAssignment[index]
            elif isInIndices(index,self.t2.indices):
                t2Indices[index] = indexAssignment[index]
            else:
                raise TensorError('Unknown index passed to ProductTensor')

        t1Eval = self.t1.evaluate(t1Indices,debug=debug)
        t2Eval = self.t2.evaluate(t2Indices,debug=debug)

        if debug:
            return t1Eval +'*'+t2Eval
        else: 
            return t1Eval * t2Eval
    def commaDerivative(self,index):
        """We evaluate the comma derivative"""
        return self.t1*self.t2.commaDerivative(index) + self.t2*self.t1.commaDerivative(index)

#The contraction tensor
class ContractionTensor(Tensor):
    """Deals with contraction of indices"""
    def __init__(self,tensor,contractionIndices,cSystem):
        """Deals with contraction.
        Args:
            tensor is the tensor to contract.
            contractionIndices is a list containing the two indices to contract
        """
        if len(contractionIndices) != 2:
            raise TensorError('Can only contract on two indices')
        if not validIndices(contractionIndices):
            raise TensorError('Contraction indices are invalid')
        for index in contractionIndices:
            if index not in tensor.indices:
                raise TensorError('Contraction index '+ index + ' is not present')
        #We get what indices we're exposing
        indexList = []

        for index in tensor.indices:
            if not isInIndices(index,contractionIndices):
                indexList.append(index)

        name = 'CONTRACTION('+contractionIndices[0] + ',' + contractionIndices[1]+')_' + tensor.name

        self.t = tensor
        self.contractionIndices = contractionIndices

        super().__init__(name,indexList,cSystem)

    def evaluate(self,indexAssignment,debug=False):
        """This is where the actual contraction takes place"""
        evals = []
        for i in range(self.cSystem.dim):
            passThrough = copy.deepcopy(indexAssignment)
            for index in self.contractionIndices:
                passThrough[index.lower()]=i

            evals.append(self.t.evaluate(passThrough,debug=debug))

        if debug:
            retVal = evals[0]
            for i in range(1,len(evals)):
                retVal = retVal + '+' + evals[i]
        else:
            retVal = sum(evals)

        return retVal



    def commaDerivative(self,index):
        """We compute the comma derivative. We need to remap if it involves our contraction indices"""
        copyIndices = copy.deepcopy(self.indices)
        copyIndices.extend(self.contractionIndices)

        if isInIndices(index,copyIndices):
            newIndex = getNextAvailableIndex(copyIndices)

            deriv = ContractionTensor(self.t[newIndex],self.contractionIndices,self.cSystem)
            return RemapTensor(deriv,newIndex,index,self.cSystem)
        else:
            return ContractionTensor(self.t[index],self.contractionIndices,self.cSystem)



#Data tensors
#The Constant tensor
class ConstantTensor(Tensor):
    """This represents a constant, i.e. if we want to do 1*tensor etc. """
    def __init__(self,value,cSystem,indexList = []):
        self.value = value
        super().__init__(str(value),indexList,cSystem)

    def evaluate(self,indexAssignment,debug=False):
        if debug:
            return str(self.value)
        else:
            return self.value

    def commaDerivative(self,index):
        self._validateNewIndex(index)

        newIndexList = copy.deepcopy(self.indices)
        newIndexList.append(index)

        return ConstantTensor(0,self.cSystem,indexList=newIndexList)

#Matrix of numbers tensor
class MatrixTensor(Tensor):
    """This represents a tensor that is a matrix of values.

    Args: 
        mat: is the numpty array style object
        indexList: is the list of indices
        cSystem: is the coordinate system
    """
    def __init__(self,name,mat,indexList,cSystem):
        if len(mat.shape) != len(indexList):
            raise TensorError('Number indices does not match matrix dimensions!')
        for l in mat.shape:
            if l!= cSystem.dim:
                raise TensorError('Mismatching spacial dimensions,' + str(cSystem.dim) + ', and matrix dimenstions, '+str(l))
        self.mat = mat

        super().__init__(name,indexList,cSystem)

    def evaluate(self,indexAssignment,debug=False):
        """Evaluate the matrix"""
        evalPos = []
        evalMat = self.mat
        for index in self.indices:
            evalPos.append(indexAssignment[index.lower()])

        if debug:
            retVal = self.name
            for i in range(len(self.indices)):
                retVal = retVal + '_' + self.indices[i] + str(evalPos[i])
        else:
            for a in evalPos:
                evalMat = evalMat[a] #This coudl be done must better, apparently mat[[0,1]] != mat[0,1]
            retVal = evalMat
        return retVal

    def commaDerivative(self,index):
        #We know the index is unique becuase of the checks in the Tensor base class
        #TODO: we might be able to apply derivatives to the matrix components
        raise TensorError('This is not a differentiable tensor')


#Matrix that can handle differentiation
class DifferentiableMatrixTensor(Tensor):
    """A tensor, reperesented by a matrix of objects that can be differentited
    Args:
        name : name of the tensor
        objMat :array of objects
        indexList: list of indices used to reference the array
        diffIndexList: list of indices representing differentiation
        cSystem: the coordinate system object
    TODO: there may be issues with  scalars here.... This needs checking
    """
    def __init__(self,name,objMat,indexList,diffIndexList,cSystem):
        #We first of all validate the index list
        if not validIndices(indexList) or not validIndices(diffIndexList):
            raise TensorError('Invalid indices for DifferentiableMatrixTensor')
        for index in diffIndexList:
            if not index.islower():
                raise TensorError('Indices representing differentiation must be lower cae')
            if isInIndices(index,indexList):
                raise TensorError('Implicit contraction in construction. Contraction should alway be performed using ContractionTensor and a relevant remapping')

        #We now check the indices correctly map the array
        if objMat.shape == (1,):
            #We have a scalar so some rules apply differently
            self.isScalar = True
            if len(indexList) != 0:
                raise TensorError('A scalar matrix has been given but indices do not reflect this')
        else:
            self.isScalar = False
            if len(objMat.shape) != len(indexList): 
                raise TensorError('Indices do not provide a map to reference the array')
            for dim in objMat.shape:
                if dim != cSystem.dim:
                    raise TensorError('Array dimensions are invalid. Must have same dimension as CoordinateSystem')

        #Everything is good, we construct our combined indexList
        self.arrayIndices = copy.deepcopy(indexList)
        self.diffIndices  = copy.deepcopy(diffIndexList)

        totalIndices = copy.deepcopy(indexList)
        diffIndicesCopy = copy.deepcopy(diffIndexList)

        totalIndices.extend(diffIndicesCopy)

        self.objMat = objMat

        super().__init__(name,totalIndices,cSystem)

    def evaluate(self,indexAssignment,debug = False):
        if debug:
            #We construct the string representing the derivative
            retVal = 'd' + str(len(self.diffIndices)) + self.name
            for index in self.arrayIndices:
                retVal = retVal + '_' +  index + str(indexAssignment[index.lower()])
            for index in self.diffIndices:
                retVal = retVal + '_D' + index + str(indexAssignment[index.lower()])
        else:
            #We find the refererence element
            if self.isScalar:
                dref = self.objMat[0]
            else:
                dref = self.objMat

                for index in self.arrayIndices:
                    dref = dref[indexAssignment[index.lower()]]

            #We have the dereferenced object
            
            for index in self.diffIndices:
                indValue = indexAssignment[index]
                derivative = self.cSystem.derivMap[indValue] #We work out what derivative to apply. This remapping is due towanting to use r,theta,z when the mesh is defined z,theta

                if derivative == None:
                    dref = 0.
                    break

                dref = dref.dx(derivative)

            retVal = dref

        return retVal




    def commaDerivative(self,index):
        #We know the index is unique and lowercase because of the checks in the Tensor base class.
        newDiffIndices = copy.deepcopy(self.diffIndices)
        newDiffIndices.append(index)

        return DifferentiableMatrixTensor(self.name, self.objMat,
                                          copy.deepcopy(self.arrayIndices), newDiffIndices, self.cSystem)


class ScalarFunctionTensor(Tensor):
    """This is a tensor that handles functions of scalars. It cannot be differentiated!

    :param tensor: the tensor, must have zero free indices
    :param func: the function to apply. Note - we only apply to the real part!
    :param name: the name of the tensor
    :param cSystem: the coordinate system
    """
    def __init__(self,name, tensor, func, cSystem):
        if tensor.indices != []:
            raise TensorError('Scalar function tensors can only be made from tensors with no free indices')
        self._rootTensor = tensor
        self._func = func 

        super().__init__(name,tensor.indices,cSystem)

    def commaDerivative(self,index):
        raise TensorError('Scalar Function Tensors do not support comma derivatives')

    def evaluate(self, indexAssignment, debug=False):
        if debug:
            raise NotImplementedError('Not implemented debugging for scalar function tensors')
        else:
            #we evaluate the root tensor and then apply the function to its real and imagianry parts
            rootEval = self._rootTensor.evaluate(indexAssignment)

            if rootEval.imag != 0:
                print('ScalarFunctionTensor is being applied to imaginary quantity, current behaviour applies function to real part and imag part sepereately')
                imagPart = self._func(rootEval.imag)
            else:
                imagPart = 0
            return ComplexQuantity(self._func(rootEval.real), imagPart)


#Kroneker delta
class KroneckerDelta(Tensor):
    """We implement the kronecker delta as a seperate class (instead of producting a MatrixTensor)
    because the evaluation rules for the generalised 2p Kronecker delta are simpler than constructing
    a big 2p dimension array.
    Ars:
        indices : indices
        cSystem: coordinate system
    """
    def __init__(self,indexList,cSystem):
        #We make sure the index list is valid
        if not validIndices(indexList):
            raise TensorError('Invalid indices for Kronecker delta')

        #We cound the upper indices
        nUpper = len([index for index in indexList if index.isupper() ])
        nLower = len([index for index in indexList if index.islower() ])

        #We must have the same number of upper and lower indices
        if nUpper != nLower:
            raise TensorError('A Kronecker delta must have same number of upper and lower indices')
        #We have same number of lower and upper indices
        if nUpper == 0:
            raise TensorError('A Kronecker delta must have p>0')
        if nUpper >1:
            raise NotImplementedError('Not yet implemented generalised Kronecker delta functons')

        super().__init__('delta',indexList,cSystem)

    def evaluate(self,indexAssignment,debug=False):
        if debug:
            retVal = self.name

            for i in range(len(self.indices)):
                index = self.indices[i]

                indexValue = indexAssignment[index.lower()]

                retVal = retVal + '_' + index + str(indexValue)
        else:

            #We only support 2 indices at the moment so this is trivial
            if indexAssignment[self.indices[0].lower()]==indexAssignment[self.indices[1].lower()]:
                retVal = 1.
            else:
                retVal = 0.
        return retVal

    def commaDerivative(self,index):
        raise TensorError('Differentiation of Kronecker delta not supported')


def PermutationSymbol(indexList,cSystem):
    """Returns the permutation symbol for the coordinate system.
    Args:
        indexList - list of indices
        cSystem   - coordinate system
    """
    if len(indexList) != cSystem.dim:
        raise TensorError('Permutation symbol must have same number of indices as coordinate dimensions')

    #we construct the permuation matrix
    mat = np.zeros([cSystem.dim]*cSystem.dim)

    #We go through all the permutations
    p = Permutation(cSystem.dim-1)

    while p is not None:
        a = mat

        for i in range(cSystem.dim):
            pos = i^p

            if i != cSystem.dim-1:
                a = a[pos]
            else:
                if p.is_even:
                    a[pos] = 1
                else:
                    a[pos] = -1
        p = p.next_lex()


    return DifferentiableMatrixTensor('pert',mat,indexList,[],cSystem)





def _safeDerivative(quant,index):
    """Safely performs derivative of a quantity. e.g. in complex quantities with a missing real or imaginary part"""
    if quant is None or quant == 0:
        return quant
    else:
        return quant.dx(index)

#Helper classes for complex numbers
class ComplexQuantity():
    """Class implements complex quantities with a real and imaginary part.
    
    :param realPart: real part of the complex quantity
    :param imagPart: imag part of the complex quantity
    :param bool negativeWavenumber: (optional) whether the wavenumbers of this quantity are in the opposite sense
    :param list(ComplexQuantity) customWavenumbers: whether we use custom wave numbers (other than the ) ones given by the coordinate system
    :param CoordinateSystem cSystem: (optional) unless differentiation is required
    """
    def __init__(self,realPart,imagPart, cSystem = None, negativeWavenumber = False, customWavenumbers = None):
        self.real = realPart
        self.imag = imagPart
        self.negativeWavenumber = negativeWavenumber
        self._cSys = cSystem


        if self._cSys is not None:
            self.customWavenumbers = customWavenumbers or [None]*self._cSys.dim
        else:
            self.customWavenumbers = None

    def conjugate(self):
        """Performs complex conjugation"""
        return ComplexQuantity(self.real,-self.imag, )

    def dx(self,i):
        """Differentiates.

        If the quantity varies liek e^{A x} then it performs the multiplication.

        This is a pretty awful implementation but avoids a complete rewrite. In reality this should be done up in the tensor framework
        """
        if self._cSys is None:
            #We probably should raise an error here but we don't, instead we flag a big warnign
            return ComplexQuantity(_safeDerivative(self.real,i),_safeDerivative(self.imag,i))
        else:
            #because this passes us not the coordinate index but the FEniCS spatial index, we get the coordinate index
            coordIndex = self._cSys.derivMap.index(i)

            if self.customWavenumbers[coordIndex] is None and self._cSys.wavenumbers[coordIndex] is None:
                return ComplexQuantity(_safeDerivative(self.real,i),_safeDerivative(self.imag,i))
            else:
                if self.negativeWavenumber: fac =  1.
                else:                       fac = -1.

                if self.customWavenumbers[coordIndex] is not None:
                    return (fac*self.customWavenumbers[coordIndex])*self
                else:
                    return(fac*self._cSys.wavenumbers[coordIndex])*self



    #Algebra definitions for complex numbers

    def __add__(self,other):
        """Adds two complex numbers together"""
        if not isinstance(other,ComplexQuantity) and other!=0 and type(other) != complex:
            #We assume that the other quantity is purely real (but raise a warning. This will happen if not complex quantites are used in places)
            return ComplexQuantity(self.real + other, self.imag, cSystem = self._cSys, negativeWavenumber = self.negativeWavenumber, customWavenumbers = self.customWavenumbers )
        else:
            return ComplexQuantity(self.real + other.real, self.imag + other.imag, cSystem = self._cSys, negativeWavenumber = self.negativeWavenumber, customWavenumbers = self.customWavenumbers )

    def __neg__(self):
        """Negation of a complex number"""
        return ComplexQuantity(-self.real,-self.imag)

    def __radd__(self,other):
        return self + other

    def __sub__(self,other):
        negOther = -other
        return self + negOther

    def __mul__(self,other):
        """Perform multiplication"""
        if isinstance(other,complex) or isinstance(other,ComplexQuantity):
            #Complex multiplications
            real = self.real*other.real - self.imag*other.imag
            imag = self.real*other.imag + self.imag*other.real

        else:
            #We assume its real and that the multiplication is sensible(!)
            real = self.real*other
            imag = self.imag*other

        return ComplexQuantity(real,imag)   

    def __rmul__(self,other):
            return self*other

    def __getitem__(self, varg):
        return ComplexQuantity(self.real[varg], self.imag[varg], cSystem = self._cSys, negativeWavenumber = self.negativeWavenumber, customWavenumbers = self.customWavenumbers )

    @property
    def shape(self):
        return self.real.shape


#Coordinate systems

class CoordinateSystem(ABC):

    #Subclasses must make self.dim and self.derivMap
    def __init__(self):
        super().__init__()

    @abstractmethod
    def MetricTensor(self,indices):
        pass

    @abstractmethod
    def ChristoffelSymbol(self,indices):
        if len(indices) !=3:
            raise TensorError('The christoffel symbol must have three indices')
        if not indices[0].isupper() and not indices[1].islower() and not indices[2].islower():
            raise TensorError(' The christoffel symbol indices must be of the form I j k')

    @abstractmethod
    def MetricDeterminant(self):
        pass

    @abstractmethod
    def normal(seld,dNormal,index):
        """Converts the dolfin normal into a tensor with the given index"""
        pass

    def LeviCivita(self,indices):
        """Returns the Levi-Civita tensor for the coordinate system.
        This is made by using constructing the permutation matrix and then dividing or
        multipling by the square root of the symbol
        :param list(str) indices: list of indices for levi-Civits
        """
        lowerCase = [index.islower() for index in indices]
        upperCase = [index.isupper() for index in indices]

        allLower = all(lowerCase)
        allUpper = all(upperCase)

        if not (allLower or allUpper):
            raise TensorError('Levi-Civita must be all lower case or all upper case')

        #We are all lower or all upper case so we make the permuation symbol
        perm = PermutationSymbol(indices,self)

        if allLower:
            factor = ConstantTensor(   dolf.sqrt(self.MetricDeterminant()({})),self)
        else:
            factor = ConstantTensor(1./dolf.sqrt(self.MetricDeterminant()({})),self)

        return factor*perm


class Cartesian(CoordinateSystem):
    def __init__(self,dim=2):
        self.dim = dim
        self.derivMap = [i for i in range(dim)]
        self.name = 'cartesian'

        if dim == 2:
            self.cNames = ['x','y']

        else:
            self.cNames = ['x','y','z']

        self.wavenumbers = [None]*dim

        super().__init__()

    def MetricTensor(self,indices):
        if len(indices) != 2:
            raise TensorError('Metric tensor only has two indices')

        #Regardless of the indices, the metric tensor is always the identity matrix
        return MatrixTensor('g',ComplexQuantity(np.eye(self.dim),np.zeros([self.dim,self.dim])),indices,self)

    def ChristoffelSymbol(self,indices):
        super().ChristoffelSymbol(indices)

        dimension = [self.dim for i in range(3)]

        return MatrixTensor('GAMMA',ComplexQuantity(np.zeros(dimension),np.zeros(dimension)),indices,self)

    def MetricDeterminant(self):
        return ConstantTensor(ComplexQuantity(1.,0.),self)

    def writeXML(self,tree):
        el = super().writeXML(tree)

    def readXML(tree, dmesh):
        return Cartesian(dim = int(tree.find('dim').text))

    def normal(self, dNormal, index):
        if self.dim == 2:
            return MatrixTensor('N',np.array([ComplexQuantity(dNormal[0], 0.), ComplexQuantity(dNormal[1], 0.)]),[index], self)
        elif self.dim == 3:
            return MatrixTensor('N',np.array([ComplexQuantity(dNormal[0], 0.), ComplexQuantity(dNormal[1], 0.), ComplexQuantity(0.,0.)]),[index], self)

class Cylindrical(CoordinateSystem):
    """This is the coordinate system for cylindrical coordinates
    Args:
        r : the spatial coordinate in the radial direction 
        disableThetaDeriv: disables derivative wrt theta"""

    def __init__(self,r,disableThetaDeriv = True):

        self.dim = 3 #r,z,theta
        self.r = r + dolf.Constant(self.ROffset)
        self.name = 'cylindrical'

        self.cNames = ['r','t','z']

        if disableThetaDeriv:
            self.derivMap = [1,None,0]
        else:
            self.derivMap = [1,2,0]

        self.wavenumbers = [None]*self.dim

    ROffset = 1e-8 #The offset to use for r

    def enableThetaDerivative(self, fullCylindrical = False):
        """Enables thetat derivative.

        :param bool fullCylindrical: (optional) whether to use full  cylindrical coordiantes i.e. allow dx(2). Default is False
        """
        self.derivMap = [1,2,0]

        if not fullCylindrical:
            self.wavenumbers[1] = ComplexQuantity(0,0)

    def normal(self, dNormal, index):
        #TODO: this will break if we ever do azimuthally wavy things
        return MatrixTensor('N',np.array([ComplexQuantity(dNormal[1], 0.),
                                          ComplexQuantity(0.,0.),
                                          ComplexQuantity(dNormal[0], 0.)]),[index], self)


    def MetricTensor(self,indices):
        if len(indices) != 2:
            raise TensorError('Metric tensor only has two indices')

        #Check which metric tensor we want

        if indices[0].islower() and indices[1].islower():
            #Standard metric tensor
            mat = np.array([[1.,0.,0.],
                            [0.,self.r*self.r,0.],
                            [0.,0.,1.]])
        elif indices[0].isupper() and indices[1].isupper():
            mat = np.array([[1.,0.,0.],
                            [0.,1./(self.r*self.r),0.],
                            [0.,0.,1.]])
        else:
            raise TensorError('Mixed metric tensor must be constructed manually!')

        return MatrixTensor('g',ComplexQuantity(mat,np.zeros(mat.shape)),indices,self)

    def ChristoffelSymbol(self,indices):
        super().ChristoffelSymbol(indices)

        mat = np.array([[[0.,0.,0.],
                         [0.,-self.r,0.],
                         [0.,0.,0.]],
                        [[0.,1./self.r,0.],
                         [1./self.r,0.,0.],
                         [0.,0.,0.]],
                        [[0.,0.,0.],
                         [0.,0.,0.],
                         [0.,0.,0.]]])

        return MatrixTensor('GAMMA',ComplexQuantity(mat,np.zeros(mat.shape)),indices,self)

    def MetricDeterminant(self):
        return ConstantTensor(ComplexQuantity(self.r*self.r,0.),self)

