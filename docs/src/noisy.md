# Open quantum systems

$$\dot{\rho} = -i[H, \rho]  + \sum_k \gamma_k \big( J_k \rho J_k^\dagger - \frac{1}{2} J_k^\dagger J_k \rho - \frac{1}{2} \rho J_k^\dagger J_k \big),$$

Using the vectorization identity, $\text{vec}(A B C) = (C^T \otimes A)\text{vec}(B)$, master equation can be cast into a Schrodinger equation with an effective Hamiltonian given by,

$$H^{e} = I \otimes H - H^T \otimes I + i\sum_k \gamma_k[J_k^*\otimes J_k - \frac{1}{2} I \otimes J_k^{\dagger}J_k - \frac{1}{2}J_k^TJ_k^*\otimes I],$$

acting on the vectorized form of the density matrix.

$$H^{e}_{0} = I \otimes H_0 - H_0^T \otimes I + i\sum_k \gamma_k[J_k^*\otimes J_k - \frac{1}{2} I \otimes J_k^{\dagger}J_k - \frac{1}{2}J_k^TJ_k^*\otimes I],$$

$$H^{e}(t) = H^{e}_{0} + I \otimes H_c(t) - H_c(t)^T \otimes I$$