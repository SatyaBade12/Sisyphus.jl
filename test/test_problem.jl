bs = SpinBasis(1 // 2)
H = Hamiltonian(sigmaz(bs), [sigmax(bs), sigmay(bs)], (a, t) -> [a])
trans = StateTransform(spindown(bs) => spinup(bs))
cost = CostFunction((x, y) -> 1.0 - real(x' * y), a -> 0.0)
prob = QOCProblem(H, trans, (0.0, 1.0), cost)
init_params = [1.0]
@test_throws ArgumentError solve(prob, init_params, ADAM(0.1))
