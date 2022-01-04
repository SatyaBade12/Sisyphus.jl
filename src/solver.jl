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
    save_iters = :save_iters in keys(kwargs) ? kwargs[:save_iters] : Int64[]
    const_op = prob.hamiltonian.const_op
    ops = prob.hamiltonian.operators
    drives = prob.hamiltonian.drives
    n_coeffs = length(ops)
    n_params = length(initial_params)
    psi, n_dim = input_data(prob.transform, n_params)
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
    constraint_gradient(θ) = gradient(ps -> cost.constraints(ps), θ)[1]
    distance_gradient(x, y) = gradient((_x, _y) -> cost.distance(_x, _y), x, y)[2]
    gradients(ps, t) = jacobian((_ps, _t) -> drives(_ps, _t), ps, t)[1]
    sol = Solution(solver.initial_params)
    p = Progress(solver.n_iter)
    n_params = length(solver.initial_params)
    for i = 1:solver.n_iter
        res = DifferentialEquations.solve(
            solver.ode_prob,
            solver.alg,
            p = (
                (solver.const_op, solver.ops),
                (drives, gradients),
                sol.params,
                solver.n_dim,
            ),
            save_start = false,
            save_everystep = false;
            solver.kwargs...,
        )
        distance = evaluate_distance(cost, res.u[1], solver.target, solver.n_dim)
        constraints = cost.constraints(sol.params)
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
        Flux.Optimise.update!(solver.opt, sol.params, grads)
        next!(p, showvalues = [(:distance, distance), (:constraints, constraints)])
        GC.gc()
    end
    sol
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

function optimize(
    prob::QOCProblem{T},
    initial_params::Vector{T},
    opt;
    alg = DP5(),
    n_iter::Integer = 1,
    kwargs...,
) where {T<:Real}

    ode_prob = augmented_ode_problem(prob, initial_params)
    target = output_data(prob.transform)
    n_dim = dimension(prob.transform)
    flux_optimize(
        prob,
        initial_params,
        ode_prob,
        opt,
        alg,
        target,
        n_dim,
        n_iter,
        kwargs...,
    )

end

function optimize(
    prob::QOCProblem{T},
    initial_params::Vector{T},
    opt::Opt;
    alg = DP5(),
    n_iter::Integer = 1,
    kwargs...,
) where {T<:Real}

    ode_prob = augmented_ode_problem(prob, initial_params)
    target = output_data(prob.transform)
    n_dim = dimension(prob.transform)
    nlopt_optimize(
        prob,
        initial_params,
        ode_prob,
        opt,
        alg,
        target,
        n_dim,
        n_iter,
        kwargs...,
    )

end

function nlopt_optimize(
    qoc_prob::QOCProblem{T},
    initial_params::Vector{T},
    ode_prob::ODEProblem,
    opt,
    alg,
    target::AbstractMatrix{Complex{T}},
    n_dim::Integer,
    n_iter::Integer,
    kwargs...,
) where {T<:Real}

    H = qoc_prob.hamiltonian
    cost = qoc_prob.cost
    n_params = length(initial_params)
    opt.maxeval = n_iter
    constraint_gradient(θ) = gradient(ps -> cost.constraints(ps), θ)[1]
    gradients(ps, t) = jacobian((_ps, _t) -> H.drives(_ps, _t), ps, t)[1]
    sol = Solution(copy(initial_params), T[])

    function opt_function(x::Vector{T}, g::Vector{T})
        @inbounds for i = 1:n_params
            sol.params[i] = x[i]
        end
        res = DifferentialEquations.solve(
            ode_prob,
            alg,
            p = ((H.const_op, H.operators), (H.drives, gradients), sol.params, n_dim),
            save_start = false,
            save_everystep = false;
            kwargs...,
        )
        grads, c = evaluate_gradient(cost, res.u[1], target, n_dim, n_params)
        grads .-= constraint_gradient(x)
        if length(g) > 0
            for i = 1:n_params
                g[i] = -grads[i]
            end
        end
        push!(sol.trace, c)
        println(c)
        return -c - cost.constraints(x)
    end
    opt.min_objective = opt_function
    (minf, minx, ret) = NLopt.optimize(opt, sol.params)
    sol.params[:] = minx
    sol

end
