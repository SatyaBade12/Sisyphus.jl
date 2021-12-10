abstract type Drive end

mutable struct ParametricDrive{T<:Real} <: Drive
    
    params::Vector{T}
    coefficients::Function
    gradient::Function
    t0::T
    t1::T

    function ParametricDrive(params::Vector{T}, coefficients::Function, times::Tuple{T, T}) where T<: Real
        gradients(ps) = t -> jacobian((t, x)-> coefficients(x)(t), t, ps)[2] 
        t0, t1 = times
        n_coeffs = length(coefficients(params)(t0))
        @assert length(gradients(params)(t0)) == length(params)*n_coeffs
        new{T}(params, coefficients, gradients, t0, t1)
    end
end