"""
Abstract base class for transformations between states.
"""
abstract type Transform end

"""
    StateTransform(input::Ket{B,T}, output::Ket{B,T}) where {B, T}

Represents a state to state transformation between two pure states.
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
    UnitaryTransform(inputs::Vector{Ket{B, T}}, U::Matrix) where {B, T}

Represents a unitary transformation between two sets of states.
The tranformation can be initialized with a vector of kets and
a unitary matrix representing the desired transformation.
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


"""
    CuKet(k::Ket)

Returns a Ket with the data allocated on GPU memory.
"""
CuKet(k::Ket) = Ket(k.basis, CuVector(k.data))

"""
    cu(trans::UnitaryTransform)

Returns a unitary transformation with the kets allocated on GPU memory.
"""
function cu(trans::UnitaryTransform)
    UnitaryTransform([
        CuKet(in) => CuKet(out) for (in, out) in zip(trans.inputs, trans.outputs)
    ])
end

"""
    cu(trans::StateTransform)

Returns a state to state transformation with the kets allocated on GPU memory.
"""
cu(trans::StateTransform) = StateTransform(CuKet(trans.input) => CuKet(trans.output))

"""
    vectorize(k::Ket)

Returns a ket representing the vectorized form of the density matrix of ket `k`.
"""
function vectorize(k::Ket)
    basis = k.basis ⊗ k.basis
    N = length(k.data)
    Ket(basis, reshape(dm(k).data, N * N))
end

"""
    vectorize(trans::StateTransform)

Returns a vectorized form of the state to state transformation.
"""
vectorize(trans::StateTransform) =
    StateTransform(vectorize(trans.input) => vectorize(trans.output))

"""
    vectorize(trans::UnitaryTransform)

Returns a vectorized form of the unitary transformation.
"""
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
