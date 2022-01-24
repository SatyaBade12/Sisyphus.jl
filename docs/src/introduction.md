# Introduction

Pulse-level control is an essential component of today's quantum devices, and production of high-fidelity control protocols is necessary in the current search for quantum computational advantage. Both gradient-based and gradient-free quantum optimal control (QOC) methods have been developed, and many of them, like Krotov, GRAPE, GOAT, and CRAB are already included in popular frameworks like QuTiP. 

Numerical methods such as [GRAPE](https://qutip.org/docs/4.0.2/guide/guide-control.html) and [Krotov](https://qucontrol.github.io/krotov/v1.0.0/01_overview.html) solve the optimal control problems by representing the drive signals as piece-wise constant functions and iteratively optimize these discrete values based on the gradient of the cost function with respect to the control parameters. In these approaches, for a given set of control signals, the resulting unitary transformation is computed by first-order Trotterization of each time slice. These methods are inherently prone to discretization errors and one must sample the control signals sufficiently to obtain high fidelity protocols. A continuous parametrization of drive signals, combined with higher-order ordinary differential equation solvers are essential for numerical optimal control problems when accuracy can not be traded with computational effort see for e.g. [GOAT](https://doi.org/10.1103/physrevlett.120.150401).


## Why do we need a dedicated library for quantum optimal control?

SciML ecosystem already supports solving [optimal control problems](https://diffeqflux.sciml.ai/dev/examples/optimal_control/) via sensitivity analysis. In principle, quantum optimal control problems can be solved with this approach, however, it has two shortcomings:

a) One should express the time evolution of Schrodinger or Master equation only in real numbers i.e. by separating the real and imaginary parts of operators and kets. Otherwise, you'll run into errors while calculating gradients of cost function (loss) using AD.

b) The size of wavefunctions and/or the number of parameters in quantum optimal control problems can be large. In such cases, it is beneficial to implement the RHS of time evolution equations with Intel MKL or CUBLAS libraries. However, AD once again fails if the equations contain calls to functions not written in pure Julia (for e.g. when `mul!` points to Intel MKL library).

In `Sisyphus.jl` package we solve the above mentionned problems by separating the AD part from the time evolution. The result is a high-performance library for quantum optimal control that enables users to: describe a general quantum optimal control problem in the familiar language of [QuantumOptics.jl](https://docs.qojulia.org/), select any [ODE solver](https://diffeq.sciml.ai/dev/solvers/ode_solve/), optimizer (in [Flux.jl](https://fluxml.ai/Flux.jl/stable/training/optimisers/#Optimiser-Reference) and [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl)) and solve it fast!