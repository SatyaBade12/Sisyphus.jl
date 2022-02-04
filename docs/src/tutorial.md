# Tutorial

The main goal of this tutorial is to introduce the terminology and workflow of the package.

## Constructing a [`Hamiltonian`](@ref)

We split the Hamiltonian into time-independent and time-dependent parts. Operators constituting the Hamiltonian are represented by QuantumOptics operators. For a list of operators we submit a `drives` function that returns the corresponding list of real-valued drives that multiplies the operators respectively. Below is a construction of a simple two-level Hamiltonian with a parameterized Gaussian shaped drive

$$H(t)/\hbar\omega_0 = -\frac{1}{2}\sigma_z + \Omega(p, t)\sigma_x$$

```julia
bs = SpinBasis(1//2)
Ω(p, t) = [p[1] * exp(-p[2] * t^2) + p[3]]
H = Hamiltonian(-0.5*sigmaz(bs), [sigmax(bs)], Ω)
```

The real valued drives can be any function in Julia, as long as it is differentiable. You can find some examples showcasing the use of neural networks, piecewise constant and linear functions in the notebooks.

## Defining a [`Transform`](@ref)

### a) [`StateTransform`](@ref)
Transformation can be defined between two Kets as in the following code.

```julia
bs = SpinBasis(1//2)
trans = StateTransform(spindown(bs) => spinup(bs))
```

### b) [`UnitaryTransform`](@ref)
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
                                  [0.0 0.0 0.0 1.0]])
```

## Constructing a [`CostFunction`](@ref)

Cost function is composed of a distance function measuring the overlap of the quantum states and the optional constraints on the shape of pulses. The following code defines a cost function that measures the infidelity between quantum states and constrains the pulse to zero at initial and final times `t0` and `t1`.

```julia
(t0, t1) = (0.0, 1.0)
Ω(p, t) = [p[1] * exp(-p[2] * t^2) + p[3]]
cost = CostFunction((x, y) -> 1.0 - abs2(x' * y),
                     p -> Ω(p, t0)[1]^2 + Ω(p, t1)[1]^2)
```

The solver automatically calculates the gradient of the distance and constraints function wrt the parameters of the problem. Therefore, the `CostFunction` arguments have to be differentiable real valued functions (`Zygote` handles most functions, just try!).

## Creating and solving a [`QOCProblem`](@ref)

Once we have constructed the Hamiltonian, cost function, and target unitary transformation, we can define a quantum optimal control problem by submitting a timeframe `(t0, t1)` for the evolution along the previously mentioned objects and functions. `QOCProblem` can be solved by invoking the `solve` method that can further be customized, e.g. through the selection of an optimizer or by setting optimization hyperparameters like below.

```julia
prob = QOCProblem(H, trans, (t0, t1), cost)
sol = solve(prob, initial_params, ADAM(0.01);)
```

The [`Solution`](@ref) returned by the solver contains the optimal parameters and also the values of distance metric and constraints during the optimization process.

Optionally, you can use the following keywords while invoking the `solve` method:

a) `maxiter` allows to set the maximum number of iterations, the default value is `100`.

b) `save_iters` to save the parameters tried by the optimizer at the specified iterations (useful for visualization of optimization). By default, the [`Solution`](@ref) object returned by the solver does not contain intermediate results.


## Selecting an optimizer

User can pick any of the available optimizers from the Flux or NLopt packages, for example
```julia
using Flux.Optimise: RMSProp
sol = solve(prob, initial_params, RMSProp(0.01); maxiter=100)
```

## Selecting a differential equation solver

The ODE solver used to compute the system dynamics and gradients can be chosen with the keyword `alg`.
```julia
sol = solve(prob, initial_params, ADAM(0.01); maxiter=100, alg=DP5(), abstol=1e-6, reltol=1e-6)
```
You may select appropriate ODE solvers available in `OrdinaryDiffEq` package. By default `Sisyphus.jl` uses `Tsit5()` algorithm, we encourage you to go through the documentation of [ODE Solvers](https://diffeq.sciml.ai/stable/solvers/ode_solve/#ode_solve) and try different algorithms to identify the algorithm best suited for your problem. In addition, you can also control the solver tolerances by setting `abstol` and `reltol`.

## Optimization in the presence of noise

Optimal control problems in the presence of Lindbladian noise can be solved by converting them into an equivalent [closed system problem](noisy.md) by vectorizing the master equation. Here, we only provide tools to convert [`Hamiltonian`](@ref) and [`Transform`](@ref)s into their vectorized forms. However, it is the responsibility of the users to provide an appropriate distance measure between the two density matrices in the [`CostFunction`](@ref) (check examples) while working with the vectorized forms. For example, one can write Frobenius norm in its vectorized form as

$$\|\rho - \sigma\|_F = \sqrt{\text{Tr}(\rho - \sigma)^\dagger(\rho - \sigma)} = \sqrt{\text{vec}(\rho - \sigma)^\dagger\text{vec}(\rho - \sigma)},$$
and define a correspoding distance function as `(x, y) -> 1.0 - sqrt((x - y)' * (x - y))`.

## Solving problems on GPU

Usually, all the kets and operators in `QOCProblem` are allocated on the CPU. In order to solve the problem on a GPU, we need to move the data to GPU memory, this can be done simply with the [`cu`](@ref) function as shown below. Once the data is moved to the GPU, the problem can be solved with the `solve` method as usual

```julia
cu_prob = cu(QOCProblem(H, trans, (t0, t1), cost))
solve(cu_prob, init_params, ADAM(0.1); maxiter=100)
```

GPUs have better single precision performance, if your problem does not require double precision, it is better to use single precision to obtain results quicker (it also reduces the memory footprint). We provide [`convert`](@ref) method to automatically convert all data types to single precision,

```julia
prob = cu(convert(Float32, QOCProblem(H, trans, (t0, t1), cost)))
```

notebooks demonstrating these features can be found in the examples.

### How to choose a `CostFunction`?

We have to bear in mind that the distance function (specified in `CostFunction`) must operate on arrays allocated on GPU. Most of the time, Julia's GPU compiler automatically generates CUDA kernels behind the scenes, both for distance and it's gradient! However it fails sometimes, for example, you should use,

```julia
cost = CostFunction((x, y) -> 1.0 - real(sum(conj(x) .* y))
```

instead of,

```julia
cost = CostFunction((x, y) -> 1.0 - real(x' * y)
```

eventhough they are equivalent. Solving this in general is beyond the scope of `Sisyphus.jl`.

