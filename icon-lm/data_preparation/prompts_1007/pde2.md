Generate captions for a partial differential equation (PDE).
---
Here are some example captions:
A PDE describing the dynamics of $u(x)$ and $c(x)$, where $c(x)$ affects the linear reaction.
$-\lambda d^2u/dx^2 + c(x) \cdot u = a$.
$- 0.001 * d^2u/dx^2 + c(x) \cdot u = a$, with boundary conditions $u(0) = coeff_ul$ and $u(1) = coeff_ur$, where we are also given $a = 0.002$, $coeff_ul = 0.003$, and $coeff_ur = 0.004$.
---
Now please design two groups of captions based on the above examples. 
---
In the first group, you can use human language or tell the form of the equation with parameters, but do not tell the value of the parameters. For example:
A PDE describing the dynamics of $u(x)$ and $c(x)$, where $c(x)$ affects the linear reaction.
The second derivative of $u$ with respect to $x$, scaled by $-\lambda$, added to the product of $c(x)$ and $u$, equals a constant $a$.
---
In the group 2, you should tell all the values of the parameters $\lambda$, $a$, $coeff_ur$ and $coeff_ul$. We give that $\lambda = 0.001$, $a = 0.002$, $coeff_ul = 0.003$, and $coeff_ur = 0.004$. In the expression, you should specify the parameters values. The following examples are good:
$- 0.001 \cdot d^2u/dx^2 + c(x) \cdot u = 0.002$, with boundary conditions $u(0) = 0.003$ and $u(1) = 0.04$.
The partial differential equation is $\lambda \cdot d^2u/dx^2 + c(x) \cdot u = a$, with boundary conditions $u(0) = 0.003$ and $u(1) = 0.004$, where we are also given $\lambda = 0.001$, $a = 0.002$.
