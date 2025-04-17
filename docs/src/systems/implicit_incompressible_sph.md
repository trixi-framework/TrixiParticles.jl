# [Implicit Incompressible SPH](@id iisph)

Implicit Incompressible SPH as introduced by [M. Ihmsen](@cite IHMSEN et al). This schemes relies on computing the pressure values, by iteratively solving a linear system with a relaxed jacobi system, to resolve the particles density deviation from the rest density. This method uses a linear system $Ax=b$ where the pressure values that are used for the pressure acceleration are the unknown variable $x$.  
It does not use a state equation to generate pressure values like the [weakly compressible SPH scheme](weakly_compressible_sph.md).


In order to get the formulation for the linear system we start by discretizing the Continuity equation 

```math
\frac{D\rho}{Dt} = - \rho \nabla \cdot v
```

For a particle $i$ a forward difference method is used to discretizie the left hand side of the equation to get

```math
\frac{\rho_i(t+ \Delta t - \rho_i(t))}{\Delta t}.
``` 

The right hand side gets discretized by the difference formulation of the SPH discretitzation of the gradient

```math
\nabla \cdot \textbf{v}_i = \frac{1}{\rho_i} \sum_j m_j \textbf{v}_{ij} \nabla W_{ij},
```
where $\textbf{v}_{ij}$ is equal to $\textbf{v}_i - \textbf{v}_j$.

So all in all the following discretized version of the continuity equation for a particle $i$ is achieved:

```math
\frac{\rho_i(t + \Delta t) - \rho_i(t)}{\Delta t} = \sum_j m_j \textbf{v}_{ij}(t+\Delta t) \nabla W_{ij}
```

For this implicit formular  the unknown velocities $\textbf{v}_{ij}(t + \Delta t)$ are needed, which depend on the unknown pressure values $p(t+\Delta t)$.
They are given by adding all pressure and non-pressure acceleartions to the current veloctiy:

```math
\textbf{v}_i(t + \Delta t) = \textbf{v}_i(t) * \frac{\textbf{F}_i^{adv}(t) + \textbf{F}_i^p(t)}{m_i}
```

$\textbf{F}_i^{adv}$ are all non-pressure forces such as gravity, viscosiy, surface tension and more, while $\textbf{F}_i^p(t)$ are the pressure forces. 

Note that the IISPH is an incompressible fluid system, which means that the density of the fluid does not change over the time. By assuming that we have a fixed density value (the reference density $\rho_0$) for all fluid particle over the whole time of the simulation, we want to have that the density value at the next time step $\rho_i(t + \Delta t)$ is also this rest density. So we can plug in $\rho_0$ for $\rho_i(t + \Delta t)$ in the formula above. 

The goal is to compute the pressure values to get the pressure acceleration that is needed to achieve the rest density for each particle in the next time step. At the moment these pressure values are unknown, but all the non-pressure forces are known in $t$.
Therefore a predicted density gets calculated by an predicted velocity 
```math
\textbf{v}_i^{adv}= \textbf{v}_i(t) + \Delta t \frac{\textbf{F}_i^{adv}(t)}{m_i},
```
which depends only on the non-pressure forces $\textbf{F}_i^{adv}$: 

```math
\rho_i^{adv} = \rho_i(t) + \Delta t \sum_j m_j \textbf{v}_{ij}^{adv} \nabla W_{ij}(t)
```

To achieve the rest density the unknown pressure forces need to resolve the compression through the non-pressure forces, that means that they have to resolve the deviation of the predicted density and the rest density. 

Therefore following equation needs to be fulfilled:
```math
\Delta t ^2 \sum_j m_j \left(  \frac{\textbf{F}\_i^p(t)}{m_i} - \frac{\textbf{F}\_j^p(t)}{m_j} \right) \nabla W_{ij}(t) = \rho_0 - \rho_i^{adv}
```

This formula comes from plugging in $\rho_0$ for $\rho(t+\Delta t)$ in the dicretizied Continuity equation and by using the above definitions for $\textbf{v}_{ij}(t+\Delta t)$ and $\rho_i^{adv}$.

The unknown pressure force is given by 

```math
\textbf{F}_i^p(t) = m_i * \nabla p_i = m_i \sum_j m_j \left( \frac{p_i(t)}{\rho_i^2(t)} - \frac{p_j(t)}{\rho_j^2(t)} \right) \nabla W_{ij}
```

If you fill in this definition in the equation, you get a linear system $\textbf{A}(t) \textbf{p}(t) = \textbf{b}(t)$ with one equation and one unknown pressure value per particle

```math
\sum_j a_{ij} p_i = b_i = \rho_0 - \rho_i^{adv}
```

This linear system needs to be solved in order to get the pressure values. This gets be done by using a relaxed jacobi scheme. 
This is a iterative nummerical method to solve a linear system $Ax=b$. In each iteration the new values of the variable $x$ get computed by the following formular

```math 
x_i^{(k+1)} = (1-\omega) x_i^{(k)} + \omega \left( \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)\right)
```

In the case of the linear system for the pressure values the formula is

```math
p_i^{l+1} = (1-\omega) p_i^l + \omega \frac{\rho_0 - \rho_i^{adv} \sum_{j \neq i} a_{ij}p_j^l}{a{ii}}
```

Therefore the diagonal elements $a_{ii}$ and the sum $\sum_{j \neq i} a_{ij}p_j^l$ need to be determined. 
This can be done efficently by seperating the formula for the pressure force acceleration into a summantor which describes the displacement of particle i due to the pressure value of particle i, and one summantor which describes the displacement du to the pressure values of the neighboring particles  

```math
\Delta t^2 \frac{\textbf{F}_i^p}{m_i} = -\Delta t^2 \sum_j m_j \left( \frac{p_i}{\rho_i^2} + \frac{p_j}{\rho_j^2} \right)\nabla W_{ij} = \left( - \Delta t^2 \sum_j \frac{m_j}{\rho_i^2} \nabla W_{ij} \right) p_i + \sum_j - \Delta t^2 \frac{m_j}{\rho_j^2} \nabla W_{ij}p_j
```

The $d_{ii}p_i$ value describes the displacement of particle i because of the pressure value $p_i$ and $d_{ij}p_j$ describes the influence from the neighboring particles $p_j$.
Using this new values the linear system can be rewritten as

```math
\rho_0 - \rho_i^{adv} = \sum_j m_j \left( d_{ii}p_i + \sum_j d_{ij}p_j - d_{jj}p_j - \sum_k d_{jk}p_k \right) \nabla W_{ij}
```

where stands for the neighbor particles of the neigbor particle from.
So the sum over the neighbors $p_j$ also includes the pressure values $p_i$ since $i$ is a neighbor of $j$
To seperate this sum it can be written as 

```math 
\sum_k d_{jk} p_k = \sum_{k \neq i} d_{jk} p_k + d_{ji} p_i
```

With this seperation the equation for the linear system can again be rewritten as 

```math 
\rho_0 - \rho_i^{adv} = p_i \sum_j m_j ( d_{ii} - d_{ij})\nabla W_{ij}  + \sum_j m_j \left ( \sum_j d_{ij} p_j - d_{jj} p_j - \sum_{k \neq i} d_{jk}p_k \right) \nabla W_{ij}
```

In this forumlation all coefficients which are getting multiplied with the pressure value $p_i$ are seperated from the other, and the values for the diagonal elements $a_{ii}$ can be defined as these coeffincients

```math
a_{ii} = \sum_j m_j ( d_{ii} - d_{ij})\nabla W_{ij}
```

The other term of the equation is the influence of the other pressure values $p_j$. So the final relaxed jacobi iteration looks like this 

```math
p_i^{l+1} = (1 - \omega) p_i^{l} + \omega \frac{1}{a_{ii}} \left( \rho_0 -\rho_i^{adv} \sum_j m_j \left( \sum_j d_{ij} p_j^l - d_{jj} p_j^l - \sum_{k \neq i} d_{jk} p_k^l \right) \nabla W_{ij} \right).
```

Note that the pressure value of a particle $i$ depends only on its own current pressure value, the pressure values of his neighboring particles, and the neighbor of the neighboring particle. That means that the most entries of the matrix are zero. 

The diagonal elements of the matrix get computed and stored at the beginning of the simulation step and never changes during the relaxed jacobi iterations. The same holds for the $d_{ii}$ values. the coefficients $d_{ij}$ get computed to calculate the $a_{ii}$. During the jacobi iterations, two loops over all particles are being done. the first is to update the values for $\sum_j d_{ij}$ and the second is to compute the updated pressure values $p_i^{l+1}$.

The final pressure update follows the given formula of the relaxed jacobi scheme, but with two exceptions. 
First of all someone can choose if he want to avoid negative pressure values by acitivating pressure clamping. Then the final pressure update would be

```math
\max(0, p_i^{l+1} = (1-\omega) p_i^l + \omega \frac{\rho_0 - \rho_i{adv} \sum_{j \neq i} a_{ij}p_j^l}{a{ii}}
```

The secound exception is to avoid the pressure update for too small diagonal elements. The simulation gets instabil if the diagonal elements $a_{ii}$ are getting to small or even zero. This can happen if a particle is isolated and has no influence from other particle around them, or just by accident because the influences are canceling each other and the term just gets zero. 
In this case the updated pressure value will be set to zero. 
There are also other options, like setting $a_{ii}$ to the threshold value is its beneath and then updare with the known forumla, or just don't update the pressure value at all, and continue with the old value. By setting the pressure value to zero, the numerical error through this can not be so big to mess up a whole simulation, as long as it doesn't happens for too many particles. 


## Boundary Handling

The previously mentioned theory only considered fluid particle interactions but didn't consider how fluid particles interacht with boundary particles. 
For this case a few changes to the above formulas have to be done. 
Firs of all the discreitzed version of the continuity equation changes in case that the neighboring particle is a boundary particle. From know on represents neighboring boundary particles and represents neighboring fluid particles. 
Then the discretizied continuity equation for boundary particles looks like this.

```math
\frac{\rho_i(t + \Delta t) - \rho_i(t)}{\Delta t} = \sum_j m_j \textbf{v}_{ij}(t+\Delta t) \nabla W_{ij} + \sum_b m_b \textbf{v}_{ib}(t+\Delta t) \nabla W_{ib}
```

But because of the fact that the boundary particles do not have an own velocity the difference between the fluid particles velocity and the boundary particles velocity is just equal to the fluid particles velocity $\textbf{v}_{ib}(t+\Delta t) = \textbf{v}_{i}(t+\Delta t)$
The same goes for the predicted density

```math
\rho_i^{adv} = \rho_i (t) + \Delta t \sum_j m_j \textbf{v}_{ij}^{adv} \nabla W_{ij}(t) + \Delta t \sum_b m_b \textbf{v}_{i}^{adv} \nabla W_{ib}(t)
```

and also for the reulting forumla for the linear system. 

```math
\Delta t^2 \sum_j m_j \left(  \frac{\textbf{F}_i^p(t)}{m_i} - \frac{\textbf{F}_j^p(t)}{m_j} \right) \nabla W_{ij} + \Delta t^2 \sum_b m_b \frac{\textbf{F}_i^p(t)}{m_i} \nabla W_{ib} = \rho_0 - \rho_i^{adv}
```

Note that because we don't have velocities for the boundary particles the fluid particles pressure acceleration also does not depend directly on the pressure forces of the boundary particles. 
But the pressure forces of the fluid particles itself and the neighboring fluid particles depend on the pressure values of all their neighboring particles, which can also be boundary particles. Because of that the pressure value of depends indirectly on the pressure values of the neighboring boundary particles

```math
\textbf{F}_i^p(t) = -m_i \sum_j \left( \frac{p_i(t)}{\rho_i^2(t)} + \frac{p_j(t)}{\rho_j^2(t)} \right) \nabla W_{ij}(t) -m_i \sum_b \left( \frac{p_i(t)}{\rho_i^2(t)} + \frac{p_b(t)}{\rho_j^2(t)} \right) \nabla W_{ib}(t)
````

From this point on the determination of the necessary coefficients for the jacobi scheme, $d_{ii}, d_{ij}$ , ... , depend on the boundary density calculator that is used for the boundary model. 


### Pressure Mirroring
In case of using pressure mirroring the pressure value $p_j$ `s nothing else but the pressure value of the fluid particle $p_i$. 
So the coefficient which describes the influence of the own pressure value $p_i$ from particle $i$ also needs to consider the boundary particles. 
Therefore the forumla for the calculation from the $d_{ii}$ values needs to be adjusted to

```math
d_{ii} = -\Delta t^2 \sum_j ˜frac{m_j}{\rho_i^2} \nabla W_{ij} - \Delta t^2 \sum_b \frac{m_b}{\rho_i^2} \nabla W_{ib}
```

and the relaxed jacobi iteration for pressure mirroring looks like this

```math 
p_i^{l+1} = (1 - \omega) p_i^l + \omega \frac{1}{a_{ii}} \left \rho_0 - \rho_i^{adv} - \sum_j m_j \left( \sum_j d_{ij} p_j^l - d_{jj}p_j^l - \sum_{k \neq i} d_{jk} p_k^l \right) \nabla W_{ij} - \sum_b m_b \sum_j d_{ij} p_j^l \nabla W_{ij} \right)
```

### Pressure Zeroing
When Pressure Zeroing is going to be used, then the pressure value $p_j$ becomes zero and just vanishes. So the boundary particles do not have a influence on the pressure forces for particle$and the calculation for the $d_{ii}$ - coefficient remians unchanged

```math
d_{ii} = -\Delta t^2 \sum_j \frac{m_j}{\rho_i^2} \nabla W_{ij}
```

The relaxed jacobi iteration is also the same like the one above for the pressure mirroring.

```math
p_i^{l+1} = (1 - \omega) p_i^l + \omega \frac{1}{a_{ii}} \left( \rho_0 - \rho_i^{adv} - \sum_j m_j \left( \sum_j d_{ij} p_j^l - d_{jj}p_j^l - \sum_{k \neq i} d_{jk} p_k^l \right) \nabla W_{ij} - \sum_b m_b \sum_j d_{ij} p_j^l \nabla W_{ij} \right)
```












 


$$@autodocs
Modules = [TrixiParticles]
Pages = [joinpath("schemes", "fluid", "implicit_incompressible_sph", "system.jl")]$$