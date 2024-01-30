# WENO scheme for 1D scalar conservation laws and Euler equations

The code follows [Essentially Non-Oscillatory and Weighted Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws](https://ntrs.nasa.gov/citations/19980007543).

## 1D scalar conservation laws



Consider the scalar conservation law: $u_t + f(u)_x = 0$.

Given the cell average $U_i$ with $i$ as the cell index, WENO iteration is as follows:

- Reconstruct the value at grids $i\pm 1/2$ with $U_{i-2:i}, U_{i-1:i+1}, U_{i:i+2}$, denote as $v^{(0)}_{i\pm 1/2}, v^{(1)}_{i\pm 1/2}, v^{(2)}_{i\pm 1/2}$.
- For smooth cases, the above reconstructions are third-order accurate. However, if there are discontinuities in cells, oscillations occur for some of them.
- Therefore we reconstruct the value at grids again with the weighted average of $v^{(0)}_{i\pm 1/2}, v^{(1)}_{i\pm 1/2}, v^{(2)}_{i\pm 1/2}$. In the report, the weights are designed based on the smooth indicators. We denote the final reconstruction as $v^+_{i + 1/2}$ and $v^-_{i - 1/2}$, where $+$ and $-$ denote the right and left side of the interface.
    - For smooth cases, $v^+_{i + 1/2}$ and $v^-_{i - 1/2}$ are fifth-order accurate. 
    - For discontinuous cases, $v^+_{i + 1/2}$ and $v^-_{i - 1/2}$ remove oscillations.
- Given $v^+_{i + 1/2}$, $v^-_{i + 1/2}$, we can calculate the numerical flux
$$\hat{f}_{i+1/2} = h(v^-_{i+1/2}, v^+_{i+1/2})$$
- For simplicity, we use Lax-Friedrichs flux 
$$h(v^-, v^+) = \frac{1}{2}[f(v^-) + f(v^+) - \alpha (v^+ - v^-)]$$
- In the end, we update the cell average with the numerical flux. The semi-discrite scheme is
    $$ \frac{dU_i}{dt} = -\frac{1}{\Delta x}(\hat{f}_{i+1/2} - \hat{f}_{i-1/2})$$
- Apply Fourth-order Runge-Kutta method for time integration.

There is a subtle point about $\alpha$. In principle, $\alpha$ can be selected locally. However, to reduce computational cost, in this scalar case, we use a global $\alpha$ for all cells at all time:
$$\alpha = \max_{u\in[\min(U_{init}), \max(U_{init})]} |f'(u)|$$
The range of $u$ should not exceed the range of $U_{init}$ in this case.
In practice, I calculated $|f'(u)|$ on a fine grid (100 grids) and take the maximum value, adding a small redundancy (0.1) to ensure stability.


As a validation, I showed the $L_{\infty}$ error of reconstruction $v^+_{i + 1/2}$ and $v^-_{i - 1/2}$, as well as the right-hand-side of the semi-discrete scheme, with function $u = \sin(2 \pi x)$, with $f = 0.5 u^2$ (Burger's equation). We can see the log-log plot of the error is linear, parallel to $1/N^5$, which means the error is $O(\Delta x^5)$. As $N$ goes larger than 1000, the error plateaus, which could be due to the machine precision.

![weno_validation](./weno_reconstruct.png)

I also show the results of Burger's equation with step initial condition, and periodic boundary condition. I didn't see any oscillations in my results, including this case.

![weno_burger](./burgers_b0.50.png)

## 1D Euler equations

I used component-wise WENO with Lax-Friedrichs flux, since it's good enough and easy to implement. We just apply the above WENO scheme to each component of the Euler equations.

But the selection of $\alpha$ is a bit tricky. Since the momentum is not conserved, we cannot use the same $\alpha$ for all time. Instead, we update $\alpha$ for all the cells in each time step, in the following way:

$$\alpha = \max_{i,j} |\lambda_j(U_i)|$$
Here $U$ is the solution vector in the last time step, $i$ is the cell index, $j$ is the eigenvalue index, with $\lambda_1 = u - c$, $\lambda_2 = u$, $\lambda_3 = u + c$, where $u$ is the velocity, $c$ is the sound speed.

Such step-wise $\alpha$ update is the same as in [this implementation](https://www.mathworks.com/matlabcentral/fileexchange/56905-weighted-essentially-non-oscillatory-weno-scheme-for-euler), except that I also add a small redundancy (0.1) in my practice.

I showed the results of Sod's shock tube problem, with $N = 400$ grids, $dt = 0.0001$, $\gamma = 1.4$. Iterate 1000 steps.

![weno_euler](./euler_gamma_1.40.png)


## Time Cost
This code is highly vectorized and GPU-accelerated. With one NVIDIA RTX4090, solving the above Sod's shock tube problem (iterate 1000 steps) with 500 different initial conditions takes about 2 seconds.