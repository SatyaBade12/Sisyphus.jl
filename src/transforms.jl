"""
Abstract base class for all transformations between the states.
"""
abstract type Transform end

"""
    StateTransform(input, output)
    StateTransform(p)

Represents a transformation between two pure states in a Ket form.
"""
struct StateTransform <: Transform

    basis::Basis
    input::Ket
    output::Ket

    StateTransform(input::Ket{B,T}, output::Ket{B,T}) where {B,T} =
        new(input.basis, input, output)
    StateTransform(p::Pair{<:Ket,<:Ket}) = StateTransform(p[1], p[2])
    StateTransform(input::Ket, output::Ket) = throw(IncompatibleBases())

end

"""
    UnitaryTransform(input, output)
    UnitaryTransform(inputs, U)

Represents a transformation between two pure states in a Ket form,
or is given by a unitary matrix acting on a set of input states.
"""
mutable struct UnitaryTransform <: Transform

    basis::Basis
    inputs::Vector{Ket}
    outputs::Vector{Ket}

    UnitaryTransform(p::Pair{<:Ket,<:Ket}) = UnitaryTransform(p[1], p[2])
    UnitaryTransform(input::Ket{B,T}, output::Ket{B,T}) where {B,T} =
        new(input.basis, [input], [output])
    UnitaryTransform(input::Ket, output::Ket) = throw(IncompatibleBases())

    function UnitaryTransform(inputs::Vector{Ket{B,T}}, U::Matrix) where {B,T}
        size(U)[1] == size(U)[2] || throw(DimensionMismatch("U is not a square matrix"))
        length(inputs) == size(U)[1] ||
            throw(DimensionMismatch("Matrix dimensions do not correspond to the inputs"))
        U * U' ≈ one(U) || throw(ArgumentError("Matrix is not unitary"))

        outputs = Ket{B,T}[]
        for r in eachrow(U)
            output = 0.0 * inputs[1]
            for (j, e) in enumerate(r)
                output += e * inputs[j]
            end
            push!(outputs, output)
        end
        tr = UnitaryTransform(inputs[1].basis)
        for (in, out) in zip(inputs, outputs)
            tr += in => out
        end
        tr
    end

    UnitaryTransform(inputs::Vector{<:Ket}, U::Matrix) = throw(IncompatibleBases())
    UnitaryTransform(bs::Basis) = new(bs, Vector{Ket}[], Vector{Ket}[])
end

Base.:+(a::UnitaryTransform, p::Pair{<:Ket,<:Ket}) = throw(IncompatibleBases())

function Base.:+(a::UnitaryTransform, p::Pair{Ket{B,T},Ket{B,T}}) where {B,T}
    a.basis == p[1].basis || throw(IncompatibleBases())
    a.basis == p[2].basis || throw(IncompatibleBases())
    for k in a.inputs
        dagger(k) * p[1] == 0.0 ||
            @warn("The transformation is not unitary $(dagger(k)*p[1])")
    end
    for k in a.outputs
        dagger(k) * p[2] == 0.0 ||
            @warn("The transformation is not unitary $(dagger(k)*p[2])")
    end
    push!(a.inputs, p[1])
    push!(a.outputs, p[2])
    a
end

CuKet(k::Ket) = Ket(k.basis, CuVector(k.data))

function cu(trans::UnitaryTransform)
    inputs = [CuKet(k) for k in trans.inputs]
    outputs = [CuKet(k) for k in trans.outputs]
    UnitaryTransform([in => out for (in, out) in zip(inputs, outputs)])
end

function cu(trans::StateTransform)
    input = CuKet(trans.input)
    output = CuKet(trans.output)
    StateTransform(input => output)
end

function vectorize(k::Ket)
    basis = k.basis ⊗ k.basis
    N = length(k.data)
    Ket(basis, reshape(dm(k).data, N * N))
end

vectorize(trans::StateTransform) =
    StateTransform(vectorize(trans.input) => vectorize(trans.output))

function vectorize(trans::UnitaryTransform)
    t = UnitaryTransform(trans.basis ⊗ trans.basis)
    for (k1, k2) in zip(trans.inputs, trans.outputs)
        t += vectorize(k1) => vectorize(k2)
    end
    t
end

Base.convert(::Type{ComplexF32}, k::Ket) = Ket(k.basis, Vector{ComplexF32}(k.data))

Base.convert(::Type{ComplexF32}, trans::StateTransform) =
    StateTransform(convert(ComplexF32, trans.input) => convert(ComplexF32, trans.output))

function Base.convert(::Type{ComplexF32}, trans::UnitaryTransform)
    trans_F32 = UnitaryTransform(trans.basis)
    for (in, out) in zip(trans.inputs, trans.outputs)
        trans_F32 += convert(ComplexF32, in) => convert(ComplexF32, out)
    end
    trans_F32
end
