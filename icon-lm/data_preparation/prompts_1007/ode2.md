Generate captions for an ordinary differential equation (ODE).
---
Here are some example captions:
An ODE with a state variable $u$ and control variable $c$.
$du(t)/dt = a \cdot c(t) \cdot u(t) + b$.
$du(t)/dt = 0.002 + a \cdot c(t) \cdot u(t)$, with $a = 0.001$.
Given parameters $a = 0.001$ and $b = 0.002$, the ODE is $du(t)/dt = a \cdot c(t) \cdot u(t) + b$.
---
Now please design two groups of captions based on the above examples. 
---
In the group 1, you can use human language or tell the form of the equation with parameters, but do not tell the value of the parameters. For example:
An ODE with a state variable $u$ and control variable $c$.
The function $u(t)$ changes over time with the rate of $b + c(t) \cdot a \cdot u(t) $".
---
In the group 2, you should tell all the values of the parameters $a$ and $b$. We give that $a = 0.001$, $b = 0.002.$ In the expression, you should specify the parameters values. The following examples are good:
$du(t)/dt = a \cdot c(t) \cdot u(t) + b$, where $b = 0.002, a = 0.001$.
The differential equations is $du(t)/dt = 0.001 \cdot c(t) \cdot u(t) + 0.002$.

