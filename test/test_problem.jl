bs = SpinBasis(1 // 2)

H = Hamiltonian(sigmaz(bs), [sigmax(bs), sigmay(bs)], a -> [1.0, 2.0])
trans = StateTransform(spindown(bs) => spinup(bs))
cost = CostFunction((x, y) -> 1.0 - real(x' * y), a -> 0.0)
prob = QOCProblem(H, trans, (0.0, 1.0), cost)
@test_throws ArgumentError solve(prob, [1.0], Adam(0.1))

H = Hamiltonian(sigmaz(bs), [sigmax(bs), sigmay(bs)], (a, t) -> [1.0])
trans = StateTransform(spindown(bs) => spinup(bs))
cost = CostFunction((x, y) -> 1.0 - real(x' * y), a -> 0.0)
prob = QOCProblem(H, trans, (0.0, 1.0), cost)
@test_throws ArgumentError solve(prob, [1.0], Adam(0.1))

H = Hamiltonian(sigmaz(bs), [sigmax(bs), sigmay(bs)], (a, t) -> [1.0, 2.0])
trans = StateTransform(spindown(bs) => spinup(bs))
cost = CostFunction((x, y) -> 1.0 - x' * y)
prob = QOCProblem(H, trans, (0.0, 1.0), cost)
@test_throws ArgumentError solve(prob, [1.0], Adam(0.1))

H = Hamiltonian(sigmaz(bs), [sigmax(bs), sigmay(bs)], (a, t) -> [1.0, 2.0])
trans = StateTransform(spindown(bs) => spinup(bs))
cost = CostFunction(x -> x)
prob = QOCProblem(H, trans, (0.0, 1.0), cost)
@test_throws ArgumentError solve(prob, [1.0], Adam(0.1))

H = Hamiltonian(sigmaz(bs), [sigmax(bs), sigmay(bs)], (a, t) -> [1.0, 2.0])
trans = StateTransform(spindown(bs) => spinup(bs))
cost = CostFunction((x, y) -> 1.0 - real(x' * y), p -> p * 1.0im)
prob = QOCProblem(H, trans, (0.0, 1.0), cost)
@test_throws ArgumentError solve(prob, [1.0], Adam(0.1))
