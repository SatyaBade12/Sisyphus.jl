# Tutorial

The main goal of this tutorial is to introduce the terminology and workflow of the package.

## Constructing a Hamiltonian

We split the Hamiltonian into time-independent and time-dependent parts. Operators constituting the Hamiltonian are represented by QuantumOptics operators. For a list of operators we submit a corresponding list of real-valued drives that multiplies them respectively. Below is a construction of a simple two-level Hamiltonian with a parameterized Gaussian shaped drive

$$H(t)/\hbar\omega_0 = -\frac{1}{2}\sigma_z + \Omega(p, t)\sigma_x$$

```julia
bs = SpinBasis(1//2)
Ω(p, t) = p[1] * exp(-p[2] * t^2) + p[3]
H = Hamiltonian(-0.5*sigmaz(bs), [sigmax(bs)], [Ω])
```

## Constructing a cost function

Cost functions is composed of a distance function measuring the overlap of the quantum states and the optional constraints on the shape of pulses. The following code defines a cost function that measures the infidelity between quantum states and constrains the pulse to zero at initial and final times `t0` and `t1`.

```julia
(t0, t1) = (0.0, 1.0)
Ω(p, t) = p[1] * exp(-p[2] * t^2) + p[3]
cost = CostFunction((x, y) -> 1 - abs2(x' * y),
                     p -> Ω(p, t0)^2 + Ω(p, t0)^2)
```

## Defining a transformation

Transformation can be defined between two Kets as in the following code.

```julia
bs = SpinBasis(1//2)
trans = StateTransform(spindown(bs) => spinup(bs));
```

It can alternatively be defined on a vector of Kets by providing unitary matrix that acts on the subspace spanned by them and represents the desired unitary evolution.

```julia
bs = FockBasis(5)

states = [fockstate(bs, 0)⊗fockstate(bs, 0),
          fockstate(bs, 0)⊗fockstate(bs, 1),
          fockstate(bs, 1)⊗fockstate(bs, 0),
          fockstate(bs, 1)⊗fockstate(bs, 1)]

trans = UnitaryTransform(states, [[1.0 0.0 0.0 0.0];
                                  [0.0 1.0 1.0im 0.0]/√2;
                                  [0.0 1.0im 1.0 0.0]/√2;
                                  [0.0 0.0 0.0 1.0]]);
```

## Creating and solving a QOC problem

Once we have constructed the Hamiltonian, cost function, and target unitary transformation, we can define a quantum optimal control problem by submitting a timeframe `(t0, t1)` for the evolution along the previously mentioned objects and functions. `QOCProblem` can be solved by invoking the `solve` method that can further be customized, e.g. through the selection of an optimizer or by setting optimization hyperparameters like below.

```julia
prob = QOCProblem(H, trans, (t0, t1), cost)
sol = solve(prob, initial_params, ADAM(0.01); maxiter=100);
```

## Selecting an optimizer

## Selecting a differential equation solver

