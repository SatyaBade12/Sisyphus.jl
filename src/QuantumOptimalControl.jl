module QuantumOptimalControl

import QuantumOpticsBase
using QuantumOptics
using MKLSparse
using LinearAlgebra
using DifferentialEquations: DP5, Tsit5, solve, Vern7
using OrdinaryDiffEq: ODEProblem
import DifferentialEquations
using Flux: jacobian
using Flux
using CUDA
using CUDA.CUSPARSE
using ProgressMeter: Progress, next!
import DiffEqBase
using DataStructures
using NLopt: Opt
import NLopt

include("transforms.jl")
include("hamiltonian.jl")
include("drive.jl")
include("loss.jl")
include("problem.jl")
include("solver.jl")


export StateTransform,
    UnitaryTransform,
    Hamiltonian,
    Solution,
    CuHamiltonian,
    CuOperator,
    CuQOCProblem,
    CostFunction,
    ParametricDrive,
    QOCProblem,
    coefficients,
    gradient,
    optimize

end
