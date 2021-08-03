# tensorForm

During my PhD I wanted to solve PDEs (Navier-Stokes and RANS) in different coordinate systems. To do this more robustly, I developed some code that used
tensor calculus to manipulate the weak forms of PDEs. In case it's useful for others, I've extracted this into `tensorForm`, a Python module that implements the
mechanics of tensor calculus.

To install the code, use:

    pip install --user -e .


To check the installation and make sure nothing is horrifically broken, invoke the test suite with:

    pytest test/

## Caveats

I originally wrote this back in 2018. There may be bugs I haven't fixed, am yet to find, or have introduced when separating this from the rest of the code I 
developed. I have included some of the original tests I wrote, which should all pass.

This is a naive implementation, essentially automating hand methods process of adding, multiplying, contracting and applying derivatives of tensors. There are 
likely better libraries that will do the same thing.

Although this doesn't strictly require FEniCS to work, the way derivatives are applied (using `.dx(i)`) and the test suite assumes that FEniCS is being used. 

When using this to solve PDEs, be aware that the output will need to be converted to [physical components](https://arxiv.org/pdf/gr-qc/0105071.pdf) using the metric tensor.

The code currently requires specification of a Coordinate systems Christoffel symbol and metric tensor. In theory the Christoffel symbol could instead be 
derived directly from the metric tensor.

The documentation here is poor, the  `test_diffusionProblem` and `test_wavenumbers` cases in the test suite are probably the best way to get an understanding 
of how things work. When I have time, I will try and add examples (and maybe even improve the docs!).

## Key Concepts

With vector calculus, converting between coordinate systems can be awkward, often requiring one to lookup the definition of <img src="https://render.githubusercontent.com/render/math?math=\nabla"> in a particular coordinate system to work out what is happening to each component.

Tensor calculus deals with tensors, objects that either translate like the Jacobian (covariant) or like the inverse of the Jacobian (contravariant) when the
coordinate system is changed. Covariant tensors are denoted with lower indices, e.g. <img src="https://render.githubusercontent.com/render/math?math=u_i">, 
whilst contravariant tensors are denoted using upper indices, e.g <img src="https://render.githubusercontent.com/render/math?math=u^i">. 
Contracting a covariant tensor with a contravariant tensor will produce an invariant, a quantity that is independent of the coordinate system.

The [covariant derivative](https://mathworld.wolfram.com/CovariantDerivative.html) of a tensor takes into account the curvature of the coordinate system 
using the [Christoffel symbol of the second kind](https://mathworld.wolfram.com/ChristoffelSymboloftheSecondKind.html). `tensorForm` uses the Christoffel
symbol together with the [metric tensor](https://mathworld.wolfram.com/MetricTensor.html) to transform weak forms between different coordinate systems.
By specifying these objects, the code will automatically handle all the coordinate transforms without needing the weak form to be respecified. The definitions
of these objects can easily be found online for different coordinate systems, e.g. in [the MathWorld page for spherical coordinates](https://mathworld.wolfram.com/SphericalCoordinates.html).

## Usage

For the code to work, a coordinate system must be specified. `Cartesian` and `Cylindrical` coordinate systems are incorporated into the module. The `Cylindrical` 
coordinate system requires specification of the `radial` coordinate. This can be done using the mesh's spatial coordinates. Custom coordinate systems can be
implemented by making a subclass of the `CoordinateSystem` abstract base class.

Once a coordinate system has been created, tensors are typically created using `DifferentiableMatrixTensor`. This class allows differentiation and represents the underlying tensor components with a matrix. The following code generates a doubly covariant tensor with a cartesian coordinate system (assumed to have been 
created earlier).

    A = DifferentiableMatrixTensor('A',mat,['i','j'],[],cartesian)

Here, `mat`, is a numpy matrix containing the components of the tensor (which may be FEniCS trial or test functions). The first list gives the free indices of the tensor, with lower case representing covariant indices and upper case representing contravariant indices.  The second list contains any indices representing comma derivatives, these are typically not required in normal use and are generated automatically by the code.

We can now manipulate the tensor in the following ways:

* Multiplication (with automatic contraction between covariant and contravariant indices)

    A*B

* Addition

    A + B

* [Comma derivative](https://mathworld.wolfram.com/CommaDerivative.html) with respect to a specific index.

    A['k']

* [Covariant derivative](https://mathworld.wolfram.com/CovariantDerivative.html) with resepect to a specific index.

    A**'k'

* Evaluation with specific index values.

    A({'i' : 1, 'j' : 2})

  To evaluate invariants, an empty dict should be passed.

By writing the weak form as an invariant, the form can be easily be switched between coordinate systems. For example, to switch from cylindrical to cartesian
coordinates, the code for the diffusion problem below only needs the word `cylindrical` to be replaced with `cartesian`.

    cylindrical = t.Cylindrical(r)

    u = t.DifferentiableMatrixTensor('u',np.array([U]),[],[],cylindrical)
    v = t.DifferentiableMatrixTensor('v',np.array([V]),[],[],cylindrical)

    uj = u**'j'
    vi = v**'i'

    met = cylindrical.MetricTensor(['I','J'])
    form = uj*met*vi

    lhs = form({})*dolf.sqrt(cylindrical.MetricDeterminant()({}))*dolf.dx

Here, the metric tensor was used to generate the invariant. This is taken directly from the coordinate system class. Depending on the case of the indices provided,
the code will return the doubly covariant, doubly contravariant, or mixed form of the metric tensor. To see examples of the code in use, look at the functions
`test_wavenumbers`, and `test_diffusionProblem` in the test suite. 

## Future Work

This document and the code is a very rough first pass to extract some interesting code from my PhD and make it usable to more people. At some point when I have
time I will clean up the code a little and add more documentation on how to use it. However, this may not be for a while!
