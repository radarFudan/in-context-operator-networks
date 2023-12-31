Generate captions for an ordinary differential equation (ODE).
---
Here are some example captions:
An ODE with a state variable $u$ and control variable $c$.
$du(t)/dt = a_1 \cdot u(t) + a_2 * c(t) + a_3$.
$du(t)/dt = 0.001 \cdot u(t) + 0.002 \cdot c(t) + 0.003$.
An ODE $du(t)/dt = a_1 \cdot u(t) + a_2 \cdot c(t) + a_3$, with $a_1 = 0.001, a_2 = 0.002, a_3 = 0.003$.
---
Now please design two groups of captions based on the above examples. 
---
In the first group, you can use human language or tell the form of the equation with parameters, but do not tell the value of the parameters. For example
An ODE with a state variable $u$ and control variable $c$.
Function $u$ change over time with the rate of $a_3 + a_1 \cdot u(t) + a_2 \cdot c(t)$.
---
In the group 2, you should tell all the values of the parameters $a_1$, $a_2$, $a_3$. We give that $a_1 = 0.001$, $a_2 = 0.002$, and $a_3 = 0.003$. In the expression, you should specify the parameters values. The following examples are good:
$du(t)/dt = 0.001 * u(t) + 0.002 * c(t) + 0.003$.
Here is a dynamic system described using $du(t)/dt = a_1 \cdot u(t) + a_2 \cdot c(t) + a_3$, with $a_1 = 0.001, a_2 = 0.002, a_3 = 0.003$.
