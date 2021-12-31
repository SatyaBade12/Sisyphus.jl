function complex_vector_gradient(f::Function)
    """
    Calculates the gradient of a complex valued function of two complex 
    vectors (x and y). The gradient is evaluated wrt the second vector (y).
    The function should take 4 arguments (in the order real(x), imag(x), real(y), imag(y))
    and return a tuple (real(f), imag(f)).
    """
    @assert applicable(f, rand(4)...)
    @assert length(f(rand(4)...)) == 2

    fr(xr, xi, yr, yi) = f(xr, xi, yr, yi)[1]
    fi(xr, xi, yr, yi) = f(xr, xi, yr, yi)[2]

    dfr_yr(xr, xi, yr, yi) =
        jacobian((_xr, _xi, _yr, _yi) -> fr(_xr, _xi, _yr, _yi), xr, xi, yr, yi)[3]
    dfr_yi(xr, xi, yr, yi) =
        jacobian((_xr, _xi, _yr, _yi) -> fr(_xr, _xi, _yr, _yi), xr, xi, yr, yi)[4]

    dfi_yr(xr, xi, yr, yi) =
        jacobian((_xr, _xi, _yr, _yi) -> fi(_xr, _xi, _yr, _yi), xr, xi, yr, yi)[3]
    dfi_yi(xr, xi, yr, yi) =
        jacobian((_xr, _xi, _yr, _yi) -> fi(_xr, _xi, _yr, _yi), xr, xi, yr, yi)[4]

    (xr, xi, yr, yi) ->
        0.5 * (
            dfr_yr(xr, xi, yr, yi) +
            dfi_yi(xr, xi, yr, yi) +
            1.0im * (dfi_yr(xr, xi, yr, yi) - dfr_yi(xr, xi, yr, yi))
        )
end
