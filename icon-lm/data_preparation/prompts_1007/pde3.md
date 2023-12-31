Generate captions for a partial differential equation (PDE).
---
Here are some example captions:
A nonlinear partial differential equation with variables $u(x)$ and $c(x)$, where $c(x)$ is a part of the source term.
Nonlinear reaction-diffusion PDE expressed as $-\lambda d^2u/dx^2 + a \cdot u^3 = c(x)$.
$- \lambda d^2u/dx^2 + a \cdot u^3 = c(x)$, with $u(0) = coeff_ul$ and $u(1) = coeff_ur$, with $\lambda = 0.001$, $a = 0.002$, $coeff_ul = 0.003$, and $coeff_ur = 0.004$.

---
Now please design two groups of captions based on the above examples. 
---
In the first group, you can use human language or tell the form of the equation with parameters, but do not tell the value of the parameters. For example
A nonlinear partial differential equation with variables $u(x)$ and $c(x)$, where $c(x)$ is a part of the source term.
Nonlinear reaction-diffusion PDE $u(x)$ expressed as $-\lambda d^2u/dx^2 + a \cdot u^3 = c(x)$.
---
In the group 2, you should tell all the values of the parameters $\lambda$, $a$, $coeff_ur$ and $coeff_ul$. We give that $\lambda = 0.001$, $a = 0.002$, $coeff_ul = 0.003$, and $coeff_ur = 0.004$. In the expression, you should specify the parameters values. The following examples are good:
$- 0.001 * d^2u/dx^2 + 0.002 * \cdot u^3 = c(x)$, with $u(0) = coeff_ul$ and $u(1) = coeff_ur$", with $coeff_ul = 0.003$, and $coeff_ur = 0.004$.
$ The nonlinear PDE is $ a \cdot (u * u * u) - \lambda d^2u/dx^2 = c(x)$, with $u(0) = 0.003$ and $u(1) = 0.004$, where we are also given $\lambda = 0.001$, $a = 0.002$.
