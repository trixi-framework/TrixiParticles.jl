---
title: 'TrixiParticles.jl: Particle-based multiphysics simulation in Julia'
tags:
  - Julia
  - SPH
  - particle-based
  - multiphysics
authors:
  - name: Niklas S. Neher
    orcid: 0009-0004-2472-0923
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Erik Faulhaber
    orcid: 0000-0001-9788-5949
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Sven Berger
    orcid: 0000-0001-6083-7038
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 3
  - name: Michael Schlottke-Lakemper
    orcid: 0000-0002-3195-2536
    affiliation: 1 # TODO: @Michael: Stuttgart or Augsburg
  - name: Gregor J. Gassner
    orcid: 0000-0002-1752-1158
    affiliation: 2
affiliations:
 - name: High-Performance Computing Center Stuttgart, University of Stuttgart, Germany
   index: 1
 - name: Department of Mathematics and Computer Science, University of Cologne, Germany
   index: 2
 - name: Institute of Surface Science, Helmholtz-Zentrum hereon, Germany
   index: 3
date: 27 March 2024 # TODO
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The realm of particle-based methods is broad and depending on the method, particles are either considered as physical particles or as mathematical interpolation points.
The latter case refers to the smoothed particle hydrodynamics (SPH) which is a method to discretize partial differential equations and is developed by [@Monaghan:1977] to simulate astrophysical applications and is currently widely used to simulate e.g. fluid and structural mechanics and even heat conduction problems.
The former case refers e.g. to discrete element method (DEM) introduced by [@Cundall:1979] which models separate, discrete particles with rotational degrees of freedom. Typical applications include those involving discontinuous materials such as granular matter, bulk materials or powders.


TrixiParticles.jl is an open-source numerical simulation framework for
particle-based multiphysics simulations, implemented in Julia as part of the modern,
Julia-based Trixi framework [@trixi].
Our primary goal is to provide a user-friendly application, accessible even to
those unfamiliar with particle-based methods. To this end, TrixiParticles.jl is designed
for easy extensibility, enabling researchers to experiment with new models or methods quickly,
without the need to study and modify large parts of the code extensively.
We strive to achieve the highest possible performance without compromising on readability
and ease of use. This approach allows users not only to prototype new ideas rapidly,
but also to scale up simulations efficiently.

TODO: Summary of the rest of the paper

# Statement of need

Numerical simulations such as computational fluid dynamics, structural mechanics, thermodynamics
or magnetohydrodynamics are often simulated using mesh-based methods. A wide range of
mature and powerful software is available for these simulations (TODO: refs?). However, in certain applications,
these methods reach their limits or are not designed to solve these problems efficiently.
This is where particle-based methods offer an alternative.

The mesh-free formalism of the methods allows for easy preprocessing, making them particularly
suitable for simulating complex geometries. This also facilitates the coupling of different single
physics into a multiphysics system.

There are several open-source software e.g. specialized for SPH methods, including [@DualSPHysics], [@SplisHSPlasH] and [@SPHinXsys],
written in C++, and  [@PySPH], written in Python. These frameworks emphasize performance and are often designed for specific purposes, such as SPH methods only.
As a result, the code might suffer from unreadability, specific method limitations or it might be hard to extend new methods.

TrixiParticles.jl provides support not only for SPH methods but also for simulating or coupling other particle-based methods such as DEM.
Another aspect is that TrixiParticles.jl is not configured at runtime via a parameter file, as it is the case with other numerical simulation codes.
Instead, each simulation is configured and set up by pure Julia code.
This makes it easy for users to add custom functionality even without touching the source code.

TODO ?: As a side note, the existing code also supports GPU implementation without the need for separate code.

# Scientific background

In TrixiParticles.jl we store the particles in systems, each representing a different type of particle-based method. Those systems are able to interact with each other, where the interaction depends on the type of the system. This approach makes it easy to add new systems.

![Particles of two different systems in a simulation domain. \label{fig:systems}](systems.png){width=40%}

To illustrate this, \autoref{fig:systems} shows particles in a simulation domain. The black ones belong to system $(1)$ and the gray ones belong to system $(2)$. To calculate the forces acting on each particle, the particles need to interact with each other.
In general, the force of a particle $a$ in system $(1)$ is calculated as following
$$ f_a^{(1)} = \sum_{b \in \mathcal{N}_{(1)}^{(1)}} f_{ab}^{(1)} + \sum_{b \in \mathcal{N}_{(1)}^{(2)}} f_{ab}^{(2)} + \dots + \sum_{b\in \mathcal{N}_{(1)}^{(n)}}f_{ab}^{(n)}, $$

where $\mathcal{N}$ is the set of neighboring particles of the corresponding system denoted by the superscript and $f_{ab}$ is the interaction force of particle $a$ and neighbor particle $b$. Each neighborhood is updated before the interaction with an efficient grid neighborhood search introduced in [@Ihmsen:2010].

The interaction of those particle systems follows method specific rules. For example, the SPH method determines the acceleration of particle $a$ according to [@Monaghan:2005] with the following interaction
$$ \frac{d v_a}{d t} = -m_a \sum_b m_b \left( \frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2} \right) \nabla_a W_{ab},$$
where $v_a$ is the velocity of particle $a$ and $m_a$, $m_b$, $\rho_a$, $\rho_b$, $p_a$, $p_b$ is the mass, density and pressure of particle $a$ and $b$, respectively. The summation is over a weighting function also called kernel-function $W_{ab}$ depending on the relative distance of particle $a$ and $b$.

# Functionality

\autoref{fig:structure} depicts the basic building blocks of TrixiParticles.jl. Basically a simulation consists of a spatial discretization, left block, and of a time integration, center block. For the latter, the [SCiML ecosystem for ordinary differential equations](https://docs.sciml.ai/DiffEqDocs/latest/) (ODE) is used. The callbacks, right block, are used to communicate with the time integration interface during the simulation.

The object `Semidiscretization` couples the passed systems to one simulation and also defines the corresponding neighborhood searches for each system. The resulting ODE problem is then fed into the time integrator and is solved by an appropriate numerical time integration scheme.


![TODO: caption *Inspired by [docs Trixi](https://trixi-framework.github.io/Trixi.jl/stable/overview/#overview-semidiscretizations)* \label{fig:structure}](structure.png){width=75%}


At present, TrixiParticles.jl includes the implementation of the following systems

* Fluid Systems
    + `WeaklyCmpressibleSPHSystem` (WCSPH): Standard SPH method originally developed by [@Monaghan:1977] to simulate astrophysics applications.
    + `EntropicallyDampedSPHSystem` (EDAC): As opposed to the WCSPH scheme, which uses an equation of state, this scheme uses a pressure evolution equation to calculate the pressure which is derived by [@Clausen:2013] and adapted to SPH by [@Ramachandran:2019].

* Structure Systems
    + `TotalLagrangianSPHSystem` (TLSPH): System to simulate elastic structure where the interaction is defined with stress tensors.
    + `DEMSystem`: Discrete element system.

* Boundary Systems
    + `BoundarySPHSystem` with several boundary models where each model follows a different interaction rule.
    + `OpenBoundarySPHSystem`: System to simulate non-reflecting boundary conditions

The code presented here can be found on [GitHub](https://github.com/trixi-framework/TrixiParticles.jl),  along with a detailed [manual](https://trixi-framework.github.io/TrixiParticles.jl/stable/) explaining how to use the package. Additionally, we provide tutorials .... reproduce results.... etc TODO.

### Feature highlights

As a young project that aims to be the Julia code for particle-based simulation we expect to integrate other methods such as particle-in-cell, incompressible SPH or heat conduction (TODO: refs) with SPH and couple them among each other and even with mesh-based methods. So far, the following feature highlights have been implemented:

* Weakly compressible SPH with density diffusion [@Antuono:2010]

* Entropically damped artificial compressibility (EDAC) [@Ramachandran:2019]

* Total Lagrangian SPH and fluid-structure interaction [@O_Connor:2021]

* Discrete element method (DEM) [@Bicanic:2004], [@Cundall:1979]

* Transport-velocity formulation (TVF) [@Adami:2013]

* Intra-particle-force surface tension [@Akinci:2013]

* Non-reflecting (open) boundaries [@Lastiwka:2009]

* Efficient grid neighborhood search [@Ihmsen:2010]

* GPU support

Figure \autoref{fig:falling_sphere} illustrates an example of our simulation results. In this example, an elastic sphere modeled with Total Lagrangian SPH falls into a tank filled with water.

![Elastic sphere falling into a tank filled with water. \label{fig:falling_sphere}](falling_sphere_combined_nonstick_4k_178.png){width=50%}

TODO: Make this nicer or omit it...
The current state allows also to validate our simulation and produce quantitative results with a post-process callback. \autoref{fig:beam_y_deflection} shows simulation results of TrixiParticles.jl (on the left) and [@DualSPHysics] (on the right) compared against a reference value of [@Turek:2007]. The curves show the y-deflection of the tip of a beam oscillating under its own weight. The different colors indicate different resolutions in the simulation. The results obtained with TrixiParticles.jl match perfectly those of [@DualSPHysics].

![Comparison of TrixiParticles.jl and  [@DualSPHysics] against [@Turek:2007]: Tip y-deflection of an oscillating beam with different resolutions, where $t_s$ is the thickness of the beam and $dp$ is the particle spacing. \label{fig:beam_y_deflection}](oscillating_beam.png){width=100%}

# Citations

TODO

# Acknowledgements

TODO
The project has benefited from funding from [hereon](https://www.hereon.de/) and [HiRSE](https://www.helmholtz-hirse.de/).

# References
