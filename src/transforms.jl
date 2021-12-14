abstract type Transform end

struct StateTransform <: Transform

    basis::Basis
    input::Ket
    output::Ket

    StateTransform(input::Ket{B,T}, output::Ket{B,T}) where {B,T} =
        new(input.basis, input, output)
    StateTransform(p::Pair{Ket{B,T},Ket{B,T}}) where {B,T} = StateTransform(p[1], p[2])
    StateTransform(input::Ket, output::Ket) = throw(IncompatibleBases)

end

mutable struct UnitaryTransform <: Transform

    basis::Basis
    inputs::Vector{Ket}
    outputs::Vector{Ket}

    UnitaryTransform(p::Pair{Ket{B,T},Ket{B,T}}) where {B,T} = UnitaryTransform(p[1], p[2])
    UnitaryTransform(input::Ket{B,T}, output::Ket{B,T}) where {B,T} =
        new(input.basis, [input], [output])
    UnitaryTransform(input::Ket, output::Ket) = throw(IncompatibleBases)

    function UnitaryTransform(inputs::Vector{Ket{B,T}}, U::Matrix) where {B,T}
        U * U' == one(U) || throw("Matrix is not unitary")
        length(inputs) == size(U)[1] ||
            throw("Matrix dimensions do not correspond to the input kets$(U*U')")
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
    function UnitaryTransform(pairs::Vector{Pair{Ket{B,T},Ket{B,T}}}) where {B,T}
        t = UnitaryTransform(pairs[1][1].basis)
        for p in pairs
            t += p
        end
        t
    end
    UnitaryTransform(inputs::Vector{Ket}, U::Matrix) = throw(IncompatibleBases)
    UnitaryTransform(bs::Basis) = new(bs, Vector{Ket}[], Vector{Ket}[])
end

Base.:+(a::UnitaryTransform, p::Pair{Ket,Ket}) = throw(IncompatibleBases)

function Base.:+(a::UnitaryTransform, p::Pair{Ket{B,T},Ket{B,T}}) where {B,T}
    a.basis == p[1].basis || throw(IncompatibleBases)
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

function CuTransform(trans::UnitaryTransform)
    inputs = [CuKet(k) for k in trans.inputs]
    outputs = [CuKet(k) for k in trans.outputs]
    UnitaryTransform([in => out for (in, out) in zip(inputs, outputs)])
end

function CuTransform(trans::StateTransform)
    input = CuKet(trans.input)
    output = CuKet(trans.output)
    StateTransform(input => output)
end
