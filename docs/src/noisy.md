# Open quantum systems

Quantum systems interacting with the environment can be modelled with the Master equation,

$$\dot{\rho} = -i[H, \rho]  + \sum_k \gamma_k \big( J_k \rho J_k^\dagger - \frac{1}{2} J_k^\dagger J_k \rho - \frac{1}{2} \rho J_k^\dagger J_k \big),$$

where $J_k$ are the jump operators and $\gamma_k$ the respective jump rates. Using the vectorization identity ($\text{vec}(A B C) = (C^T \otimes A)\text{vec}(B)$), we can cast a quantum optimal control problem in the presence of Lindbladian noise into a `QOCProblem` in the absence of noise, but with an effective Hamiltonian,

$$H^{e} = I \otimes H - H^T \otimes I + i\sum_k \gamma_k[J_k^*\otimes J_k - \frac{1}{2} I \otimes J_k^{\dagger}J_k - \frac{1}{2}J_k^TJ_k^*\otimes I],$$

acting on the vectorized form of the density matrix. This trick allows us to use the same solver to solve optimal control problems in the presence of noise.

For a general time-dependent Hamiltonian $H(t) = H_0 + H_c(t)$, the effective time-dependent Hamiltonian is,

$$H^{e}(t) = H^{e}_{0} + I \otimes H_c(t) - H_c(t)^T \otimes I,$$

where the constant part of the effective Hamiltonian is,

$$H^{e}_{0} = I \otimes H_0 - H_0^T \otimes I + i\sum_k \gamma_k[J_k^*\otimes J_k - \frac{1}{2} I \otimes J_k^{\dagger}J_k - \frac{1}{2}J_k^TJ_k^*\otimes I].$$

In `Sisyphus.jl`, we provide tools to convert [`Hamiltonian`](@ref) and [`Transform`](@ref)s into their vectorized form.