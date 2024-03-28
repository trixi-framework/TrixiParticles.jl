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
The former case refers e.g. to discrete element method (DEM) introduced by [@Bicanic:2004] which models separate, discrete particles with rotational degrees of freedom. Typical applications include those involving discontinuous materials such as granular matter, bulk materials or powders.

Numerical simulations such as computational fluid dynamics, structural mechanics, thermodynamics
or magnetohydrodynamics are often simulated using mesh-based methods. A wide range of
mature and powerful software is available for these simulations. However, in certain applications,
these methods reach their limits or are not designed to solve these problems efficiently.
This is where particle-based methods offer an alternative.

The mesh-free formalism of the methods allows for easy preprocessing, making them particularly
suitable for simulating complex geometries. This also facilitates the coupling of different single
physics into a multiphysics system.

TODO: Summary of the rest of the paper

# Statement of need

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

TODO:
"A description of how this software compares to other commonly-used packages in this research area."

| open-source software          |Programming language  |  GPU support |
|-------------------------------|----------------------|--------------|
| TrxiParticles.jl              | Julia                | yes          |
| PySPH [@PySPH]                | Python               | no           |
| DualSPHysics [@DualSPHysics]  | C++                  | yes          |
| SplisHSPlasH [@SplisHSPlasH]  | C++                  | yes          |
| SPHinXsys [@SPHinXsys]        | C++                  | yes          |


TODO Add to .bib:
- [DualSPHysics](https://github.com/DualSPHysics/DualSPHysics)
- [PySPH](https://github.com/pypr/pysph)
- [SplisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)
- [SPHinXsys](https://github.com/Xiangyu-Hu/SPHinXsys)


# Scientific background
In TrixiParticles.jl we store the particles in systems, each representing a different type of particle-based method.
The different systems are able to interact with each other, where the interaction depends on the type of the system. This approach makes it easy to add new systems.

![Particles of two different systems in a simulation domain.\label{fig:systems}](systems.png){width=35%}

Figure ? shows particles in a simulation domain. The black ones belong to system $(1)$ and the gray ones belong to system $(2)$.
To calculate the forces acting on each particle, the particles need to interact with each other.
In general, the force of a particle $a$ in system $(1)$ is calculated as following
$$ f_a^{(1)} = \sum_{b \in \mathcal{N}^{(1)}} f_{ab}^{(1)} + \sum_{b \in \mathcal{N}^{(2)}} f_{ab}^{(2)} + \dots + \sum_{b\in \mathcal{N}^{(n)}}f_{ab}^{(n)}, $$

where the superscripts denote the system, $\mathcal{N}$ is the set of neighboring particles of the corresponding system and $f_{ab}$ is the interaction force of particle $a$ and neighbor particle $b$. Each neighborhood is updated before the interaction with an efficient grid neighborhood search introduced in [@Ihmsen:2010].

The interaction of those particle systems follows method specific rules. For example, the SPH method determines the acceleration of particle $a$ basically with the following interaction
$$ \frac{d v_a}{d t} = -m_a \sum_b m_b \left( \frac{p_b}{\rho_a^2} + \frac{p_a}{\rho_a}^2 \right) \nabla_a W_{ab},$$
where $v_a$ is the velocity of particle $a$ and $m_a$, $m_b$, $\rho_a$, $\rho_b$, $p_a$, $p_b$ is the mass, density and pressure of particle $a$ and $b$, respectively. The summation is over a weighting function also called kernel-function $W_{ab}$ depending on the relative distance of particle $a$ and $b$.

TODO: make this nicer
Currently implemented systems:
Fluid Systems
- WeaklyCmpressibleSPHSystem (WCSPH): Standard SPH method originally developed by [@Monaghan:1977] to simulate astrophysics applications.
- EntropicallyDampedSPHSystem (EDAC): As opposed to the WCSPH scheme, which uses an equation of state, this scheme uses a pressure evolution equation to calculate the pressure which is derived by [@Clausen:2013] and adapted to SPH by [@Ramachandran:2019].

Strutrure Systems
- TotalLagrangianSPHSystem (TLSPH): System to simulate elastic structure where the interaction is defined with stress tensors.
- DEMSystem: Discrete element system.

Boundary Systems
- BoundarySPHSystem with several boundary models where each model follows a different interaction rule.
- OpenBoundarySPHSystem: System to simulate non-reflecting boundary conditions

TODO: Mention adaptive time-integration

# Feature highlights

Figure \autoref{fig:falling_sphere} illustrates an example of our simulation results.
In this example, an elastic sphere modeled with Total Lagrangian SPH falls into a tank filled with water.

![Elastic sphere falling into a tank filled with water.\label{fig:falling_sphere}](falling_sphere_combined_nonstick_4k_178.png){width=50%}

Currently the following feature highlights are implemented
- Weakly compressible SPH with density diffusion [@Antuono:2010]
- Entropically damped artificial compressibility (EDAC) [@Ramachandran:2019]
- Total Lagrangian SPH and fluid-structure interaction [@O_Connor:2021]
- Discrete element method (DEM) [@Bicanic:2004], [@Cundall:1979]
- Transport-velocity formulation (TVF) [@Adami:2013]
- Intra-particle-force surface tension [@Akinci:2013]
- Non-reflecting (open) boundaries [@Lastiwka2009]
- Efficient grid neighborhood search [@Ihmsen:2010]
- GPU support


As a young project that aims to be the Julia code for particle-based simulation we expect to integrate other methods
like particle-in-cell, incompressible SPH or heat conduction with SPH and couple those among each other and even with mesh-based methods.

# Citations
TODO
# Acknowledgements
TODO
The project has benefited from funding from [hereon](https://www.hereon.de/) and [HiRSE](https://www.helmholtz-hirse.de/).

# References
TODO
