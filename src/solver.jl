mutable struct Solution{T<:Real}
    params::Vector{T}
    distance_trace::Vector{T}
    constraints_trace::Vector{T}
    params_trace::Vector{Vector{T}}
    function Solution(params::Vector{T}) where {T<:Real}
        s = new{T}()
        s.params = params
        s.distance_trace = T[]
        s.constraints_trace = T[]
        s.params_trace = T[]
        s
    end
end

mutable struct AdjointSolver{T<:Real}
    initial_params::Vector{T}
    ode_prob::ODEProblem
    opt::Any
    alg::Any
    drives::Function
    gradients::Function
    const_op::AbstractMatrix{Complex{T}}
    ops::Vector{AbstractMatrix{Complex{T}}}
    target::AbstractMatrix{Complex{T}}
    cost::CostFunction
    n_dim::Integer
    n_iter::Integer
    save_iters::AbstractRange
    kwargs::Any
end

function init(prob::QOCProblem, args...; kwargs...) where {T<:Real}

    initial_params, opt = args
    alg = :alg in keys(kwargs) ? kwargs[:alg] : Tsit5()
    maxiter = :maxiter in keys(kwargs) ? kwargs[:maxiter] : 100
    save_iters = :save_iters in keys(kwargs) ? kwargs[:save_iters] : 1:-1
    const_op = prob.hamiltonian.const_op
    ops = prob.hamiltonian.operators
    drives = prob.hamiltonian.drives
    n_params = length(initial_params)
    check_compatibility(drives, length(ops), n_params)
    psi, n_dim = input_data(prob.transform, n_params)
    check_compatibility(prob.cost, n_dim, n_params)
    gradients(ps, t) = jacobian((_ps, _t) -> drives(_ps, _t), ps, t)[1]
    ode_prob = ODEProblem{true}(
        adjoint_system!,
        psi,
        prob.tspan,
        ((const_op, ops), (drives, gradients), n_dim, initial_params),
    )
    target = output_data(prob.transform)
    AdjointSolver(
        copy(initial_params),
        ode_prob,
        opt,
        alg,
        drives,
        gradients,
        const_op,
        ops,
        target,
        prob.cost,
        n_dim,
        maxiter,
        save_iters,
        kwargs,
    )
end

function solve!(solver::AdjointSolver{T}) where {T<:Real}
    drives = solver.drives
    cost = solver.cost
    constraint_gradient(θ) =
        cost.constraints == nothing ? T(0) : gradient(ps -> cost.constraints(ps), θ)[1]
    distance_gradient(x, y) = gradient((_x, _y) -> cost.distance(_x, _y), x, y)[2]
    drives_gradients(ps, t) = jacobian((_ps, _t) -> drives(_ps, _t), ps, t)[1]
    sol = Solution(solver.initial_params)
    optimize!(
        sol,
        solver.opt,
        solver,
        drives_gradients,
        constraint_gradient,
        distance_gradient,
    )
end

function adjoint_system!(
    dpsi::AbstractMatrix{Complex{T}},
    psi::AbstractMatrix{Complex{T}},
    p::Tuple{Tuple{op,Vector{op}},Tuple{Function,Function},Vector{T},Integer},
    t::T,
) where {T<:Real,op<:AbstractMatrix{Complex{T}}}

    ops, (_ct, _gt), params, n_dim = p
    n_ops = length(ops[2])
    ct = _ct(params, t)
    mul!(dpsi, ops[1], psi, -T(1)im, T(0)im)
    @inbounds for i = 1:n_ops
        mul!(dpsi, ops[2][i], psi, -T(1)im * T(ct[i]), T(1) + T(0)im)
    end
    gt = _gt(params, t)
    _psi = @view psi[:, 1:n_dim]
    @inbounds for i = 1:length(params)
        _dpsi = @view dpsi[:, n_dim*i+1:n_dim*(i+1)]
        @inbounds for j = 1:n_ops
            mul!(_dpsi, ops[2][j], _psi, -T(1)im * T(gt[j, i]), T(1) + T(0)im)
        end
    end

end

function input_data(t::UnitaryTransform, n_params::Integer)
    n_dim = length(t.inputs)
    wf_size = length(t.inputs[1])
    psi = hcat([elm.data for elm in t.inputs]...)
    psi = hcat([psi, zeros(eltype(t.inputs[1].data), wf_size, n_dim * n_params)]...)
    psi, n_dim
end

function input_data(t::StateTransform, n_params::Integer)
    n_dim = 1
    wf_size = length(t.input)
    psi = hcat([t.input.data]...)
    psi = hcat([psi, zeros(eltype(t.input.data), wf_size, n_params)]...)
    psi, n_dim
end

output_data(t::StateTransform) = hcat([t.output.data]...)
output_data(t::UnitaryTransform) = hcat([k.data for k in t.outputs]...)

function augmented_ode_problem(
    prob::QOCProblem{T},
    initial_params::Vector{T},
) where {T<:Real}
    H = prob.hamiltonian
    n_coeffs = length(H.operators)
    n_params = length(initial_params)
    psi, n_dim = input_data(prob.transform, n_params)
    gradients(ps, t) = jacobian((_ps, _t) -> H.drives(_ps, _t), ps, t)[1]
    ODEProblem{true}(
        adjoint_system!,
        psi,
        prob.tspan,
        (H.const_op, H.operators, (H.drives, gradients), n_dim, initial_params),
    )

end

function evaluate_distance(
    cost::CostFunction,
    result::AbstractMatrix{Complex{T}},
    target::AbstractMatrix{Complex{T}},
    n_dim::Integer,
) where {T<:Real}

    c = T(0)
    final_states = @view result[:, 1:n_dim]
    for (x, y) in zip(eachcol(target), eachcol(final_states))
        c += real(cost.distance(x, y) / T(n_dim))
    end
    c

end

function evaluate_gradient(
    distance_gradient::Function,
    result::AbstractMatrix{Complex{T}},
    target::AbstractMatrix{Complex{T}},
    n_dim::Integer,
    n_params::Integer,
) where {T<:Real}

    final_states = @view result[:, 1:n_dim]
    gradients = []
    @inbounds for i = 1:n_params
        res = @view result[:, n_dim*i+1:n_dim*(i+1)]
        _c = T(0)
        for (j, (x, y)) in enumerate(zip(eachcol(target), eachcol(res)))
            _c += real(distance_gradient(x, final_states[:, j])' * y)
        end
        push!(gradients, _c / T(n_dim))
    end
    gradients

end

dimension(t::UnitaryTransform) = length(t.inputs)
dimension(t::StateTransform) = 1

function optimize!(
    sol::Solution{T},
    opt, #flux optimizer
    solver::AdjointSolver{T},
    drives_gradients::Function,
    constraint_gradient::Function,
    distance_gradient::Function,
) where {T<:Real}

    p = Progress(solver.n_iter)
    n_params = length(solver.initial_params)
    for i = 1:solver.n_iter
        res = solve(
            solver.ode_prob,
            solver.alg,
            p = (
                (solver.const_op, solver.ops),
                (solver.drives, drives_gradients),
                sol.params,
                solver.n_dim,
            ),
            save_start = false,
            save_everystep = false;
            solver.kwargs...,
        )
        distance = evaluate_distance(solver.cost, res.u[1], solver.target, solver.n_dim)
        constraints =
            solver.cost.constraints == nothing ? T(0) : solver.cost.constraints(sol.params)
        push!(sol.distance_trace, distance)
        push!(sol.constraints_trace, constraints)
        grads = evaluate_gradient(
            distance_gradient,
            res.u[1],
            solver.target,
            solver.n_dim,
            n_params,
        )
        if i ∈ solver.save_iters
            push!(sol.params_trace, copy(sol.params))
        end
        grads .+= constraint_gradient(sol.params)
        Flux.Optimise.update!(opt, sol.params, grads)
        next!(p, showvalues = [(:distance, distance), (:constraints, constraints)])
        GC.gc()
    end
    sol

end

function optimize!(
    sol::Solution{T},
    opt::Opt, #NLopt optimizer
    solver::AdjointSolver{T},
    drives_gradients::Function,
    constraint_gradient::Function,
    distance_gradient::Function,
) where {T<:Real}

    if string(opt.algorithm)[2] != 'D'
        @warn "$(opt.algorithm) does not use the computed gradient"
    end

    p = Progress(solver.n_iter)
    n_params = length(solver.initial_params)
    opt.maxeval = solver.n_iter

    function opt_function(x::Vector{T}, g::Vector{T})
        @inbounds for i = 1:n_params
            sol.params[i] = x[i]
        end

        res = solve(
            solver.ode_prob,
            solver.alg,
            p = (
                (solver.const_op, solver.ops),
                (solver.drives, drives_gradients),
                sol.params,
                solver.n_dim,
            ),
            save_start = false,
            save_everystep = false;
            solver.kwargs...,
        )

        distance = evaluate_distance(solver.cost, res.u[1], solver.target, solver.n_dim)
        constraints =
            solver.cost.constraints == nothing ? T(0) : solver.cost.constraints(sol.params)
        grads = evaluate_gradient(
            distance_gradient,
            res.u[1],
            solver.target,
            solver.n_dim,
            n_params,
        )
        grads .+= constraint_gradient(sol.params)
        if length(g) > 0
            for i = 1:n_params
                g[i] = grads[i]
            end
        end
        push!(sol.distance_trace, distance)
        push!(sol.constraints_trace, constraints)
        next!(p, showvalues = [(:distance, distance), (:constraints, constraints)])
        GC.gc()
        return distance + constraints
    end
    opt.min_objective = opt_function
    (minf, minx, ret) = NLopt.optimize(opt, sol.params)
    sol.params[:] = minx
    sol

end

function check_compatibility(drives::Function, n_ops::Integer, n_params::Integer)
    if !applicable(drives, rand(n_params), rand())
        throw(ArgumentError("drives should be of the form: f(params, t)"))
    end
    if length(drives(rand(n_params), rand())) != n_ops
        throw(
            ArgumentError(
                "drives must return a vector of length equal to the number of operators",
            ),
        )
    end
end

function check_compatibility(cost::CostFunction, n_dim::Integer, n_params::Integer)
    if cost.constraints != nothing
        if !applicable(cost.constraints, rand(n_params))
            throw(
                ArgumentError(
                    "constraints in the cost function should be of the form: f(params)",
                ),
            )
        end
        if !isreal(cost.constraints(rand(n_params)))
            throw(ArgumentError("constraints function must return real value"))
        end
    end
    if !applicable(cost.distance, rand(ComplexF64, n_dim), rand(ComplexF64, n_dim))
        throw(ArgumentError("invalid distance in the cost function"))
    end
    if !isreal(cost.distance(rand(ComplexF64, n_dim), rand(ComplexF64, n_dim)))
        throw(ArgumentError("distance function must return a real value"))
    end
end

function check_compatibility(cost::CostFunction, n_dim::Integer, n_params::Integer)
    if cost.constraints != nothing
        if !applicable(cost.constraints, rand(n_params))
            throw(ArgumentError("invalid constraints in the cost function"))
        end
        if !isreal(cost.constraints(rand(n_params)))
            throw(ArgumentError("constraints must be a real valued function"))
        end
    end
    if !applicable(cost.distance, rand(n_dim), rand(n_dim))
        throw(ArgumentError("invalid distance in the cost function"))
    end
    if !isreal(cost.distance(rand(n_dim), rand(n_dim)))
        throw(ArgumentError("distance must be a real valued function"))
    end
end