---
Design two groups of captions for a mean-feild control problem.
---
In the first group, you can use human language or tell the form of the equation with parameters, but do not tell the value of the parameters. You are encouraged to use formulas and human language. For example:
Mean field control problem with density $\rho$ and terminal cost $\int g(x)\rho(1,x)dx$。
$\inf_{\rho, m}\int \frac{10m^2}{\rho} dx dt + \int g(x) \rho(1,x) dx$ s.t. $\partial_t \rho(t,x) + \nabla_x m(t,x) = 0.02 \Delta_x \rho(t,x)$ for $t \in [0,1], x \in [0,1]$ with unknown initial density $\rho_0(x)$.

In the second group, you should tell all the values of the parameters. You should specify the parameter $\rho_0(x)$ as "$\rho_0(0), \rho_0(0.1) ... \rho_0(0.9)$ = [\rho_0]", where [\rho_0] should leave to me. You are recommended to use formatted strings. For example: "Mean field control problem $\inf_{\rho, m}\int \frac{10m^2}{\rho} dx dt + \int g(x)\rho(1,x) dx$ s.t. $\partial_t \rho(t,x) + \nabla_x m(t,x) = 0.02 \Delta_x \rho(t,x)$, for $t \in [0,1], x \in [0,1],  periodic spatial boundary condition, where the initial density function \rho_0(x) satisfies $\rho_0(0), \rho_0(0.1), ..., \rho_0(0.9)$ = [\rho_0]." You should keep both objective function and the constraint given in the example.
---
Requirements:
0, You are an expert in the field of PDEs and ODEs. You have several publications in peer-reviewed journals. You are familiar with the notations and the equations.
1, In both groups, you should introduce the notations $\rho$ and $g$ or $\rho(t,x)$ and $g(x)$ 
2, You are encouraged to write the same equation in different ways, even in the same group. For example, you can either use $\partial_t \rho$ or $\frac{\partial \rho(t,x)}{\partial t}$ to represent the time derivative of $\rho$.  You can either use $g(x) * \rho(1,x)$ or $g(x) \cdot \rho(1,x)$ to represent product. 
3, Make these captions as diversified as possible, but also mathematically correct. You can reuse the example provided.
4, In group 2, for each variable, use square brackets to enclose the variable name. Each example includes both equations and the values for parameters. You should include all the variables in each example and give more accurate information compared to group 1. For example, \rho_0 should be written as [\rho_0]. Do not write ``the parameter needs to be determined'' or similar sentences. 
5, Group 1 should contain no specific values for the parameters, i.e., it should contain no square brackets.

---
In each group, using one line for each example. Do not use any format. Do not number them or use lists. Do not write ``group 1'' or related words. Do not use quotes. The answer should only contain the examples and the empty lines. Using the period sign at the end of each sentence, but do not use empty lines between examples in the same group.
---
Design and list all 20 examples for group 1 and 20 examples for group 2; First you list all 20 examples of group 1,  then use one empty line to seperate group 1 and group 2 , next list all 20 examples for group 2.