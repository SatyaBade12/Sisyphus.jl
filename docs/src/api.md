# API

## Types

```@docs
Hamiltonian
```

```@docs
Transform
```

```@docs
StateTransform
```

```@docs
UnitaryTransform
```

```@docs
CostFunction
```

```@docs
QOCProblem
```

```@docs
AdjointSolver
```

```@docs
Solution
```

## Functions

```@docs
schroedinger_dynamic
```

```@docs
master_dynamic
```

```@docs
vectorize
```

### Utilities

```@docs
heaviside
```

```@docs
interval
```

```@docs
piecewise_const_interp
```

```@docs
linear_interp
```

```@docs
cubic_spline_interp
```

## GPU

```@docs
CuKet(k::Ket)
```

```@docs
cu
```

## Single precision

```@docs
convert(::Type{Float32}, prob::QOCProblem)
```