Generate captions for a partial differential equation (PDE).
---
Here are some example captions:
A PDE with a state variable $u$ and source term $c$".
$d^2u/dx^2 = c(x)$.
$d^2u/dx^2 = c(x)$, with $u(0) = 0.001$ and $u(1) = 0.002$.
This is a differential equation, $d^2u/dx^2 = c(x)$, with $u(0) = coeff_ul$ and $u(1) = coeff_ur$, where we are also given $coeff_ul = 0.001$, and $coeff_ur = 0.002$.
---
Now please design two groups of captions based on the above examples. 
---
In the first group, you can use human language or tell the form of the equation with parameters, but do not tell the value of the parameters. For example:
A PDE with source term $c(x)$ and state $u(x)$".
The second derivative of $u(x)$ with respect to $x$ is equal to $c(x)$.
---
In the group 2, you should tell all the values of the parameters $coeff_ur$ and $coeff_ul$. We give that $coeff_ul = 0.001$, $coeff_ur = 0.002.$ In the expression, you should specify the parameters values. The following examples are good:
"$d^2u/dx^2 = c(x)$, with $u(1) = coeff_ur, u(0) = coeff_ul$, where we are also given $coeff_ul = 0.001$, and $coeff_ur = 0.002$.
The differential equation is $d^2u/dx^2 = c(x)$, with $u(0) = 0.001, u(1) = 0.002$.
