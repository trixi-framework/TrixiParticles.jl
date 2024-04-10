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

TrixiParticles.jl, part of the Trixi Project[@trixi], is an innovative Julia-based open-source framework designed for  particle-based multiphysics simulations. It aims to overcome the limitations of traditional mesh-based methods in handling complex geometries and specialized applications, such as computational fluid dynamics and structural mechanics, by offering a versatile platform for Smoothed Particle Hydrodynamics (SPH) and the Discrete Element Method (DEM), among others. Unique in its approach, TrixiParticles.jl facilitates the easy addition of new particle systems and their interactions, enhancing research flexibility. This is further complemented by its user-friendly design, which allows simulations to be configured directly with Julia code, eliminating the need for parameter files. This feature not only simplifies the integration of custom functionalities but also promotes rapid prototyping and efficient scaling of simulations, establishing TrixiParticles.jl as a robust and adaptable tool for advancing multiphysics simulations in various scientific and engineering fields.

Here, we give a brief overview of the software TrixiParticles.jl, starting with a description of the scientific background before going on to describe the functionality and usefulness in more detail.
Finally, some of the results obtained and some of the features that have been implemented so far are briefly presented.

# Statement of need

Numerical simulations such as computational fluid dynamics, structural mechanics, thermodynamics
or magnetohydrodynamics are often simulated using mesh-based methods. A wide range of
mature and powerful software is available for these simulations (TODO: cite OpenFoam etc?). However, in certain applications,
these methods reach their limits or are not designed to solve these problems efficiently (TODO: Ref to biomechanic, welding processes etc? ).
This is where particle-based methods offer an alternative.

The mesh-free formalism of the methods allows for easy preprocessing, making them particularly
suitable for simulating complex geometries. This also facilitates the coupling of different single
physics into a multiphysics system.

There are several open-source software e.g. specialized for SPH methods, including [@DualSPHysics], [@SplisHSPlasH] and [@SPHinXsys],
written in C++, and  [@PySPH], written in Python. These frameworks emphasize performance and are designed for a variety of SPH schemes.

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
$$ \frac{d v_a}{d t} = -m_a \sum_b m_b \left( \frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2} \right) \nabla_a W_{ab} + \Pi_{ab},$$
where $v_a$ is the velocity of particle $a$ and $m_a$, $m_b$, $\rho_a$, $\rho_b$, $p_a$, $p_b$ is the mass, density and pressure of particle $a$ and $b$, respectively. The last term $\Pi_{ab}$ includes dissipative terms such as artificial viscosity [@Monaghan:2005] and is scheme specific. The summation is over a weighting function also called kernel-function $W_{ab}$ depending on the relative distance of particle $a$ and $b$.

# Functionality

\autoref{fig:structure} depicts the basic building blocks of TrixiParticles.jl. A simulation essentially consists of spatial discretization (left block) and time integration (center block). For the latter, the [SCiML ecosystem for ordinary differential equations](https://docs.sciml.ai/DiffEqDocs/latest/) (ODE) is used. The callbacks (right block) are used to communicate with the time integration interface during the simulation.

The object [`Semidiscretization`](https://trixi-framework.github.io/TrixiParticles.jl/stable/general/semidiscretization/) couples the passed systems to one simulation and also creates the corresponding neighborhood searches for each system. The resulting ODE problem is then fed into the time integrator and is solved by an appropriate numerical time integration scheme.


![TODO: caption *Inspired by [docs Trixi](https://trixi-framework.github.io/Trixi.jl/stable/overview/#overview-semidiscretizations)* \label{fig:structure}](structure.png){width=100%}


At present, TrixiParticles.jl includes the implementation of the following systems

* *Fluid Systems*
    + [`WeaklyCmpressibleSPHSystem` (WCSPH)](https://trixi-framework.github.io/TrixiParticles.jl/stable/systems/weakly_compressible_sph/): Standard SPH method originally developed by [@Monaghan:1977] to simulate astrophysics applications.
    + [`EntropicallyDampedSPHSystem` (EDAC)](https://trixi-framework.github.io/TrixiParticles.jl/stable/systems/entropically_damped_sph/): As opposed to the WCSPH scheme, which uses an equation of state, this scheme uses a pressure evolution equation to calculate the pressure which is derived by [@Clausen:2013] and adapted to SPH by [@Ramachandran:2019].

* *Structure Systems*
    + [`TotalLagrangianSPHSystem` (TLSPH)](https://trixi-framework.github.io/TrixiParticles.jl/stable/systems/total_lagrangian_sph/): System to simulate elastic structure where the interaction is defined with stress tensors [@O_Connor:2021].
    + `DEMSystem`: Discrete element system [@Bicanic:2004], [@Cundall:1979].

* *Boundary Systems*
    + [`BoundarySPHSystem`](https://trixi-framework.github.io/TrixiParticles.jl/stable/systems/boundary/) with several boundary models where each model follows a different interaction rule.
    + `OpenBoundarySPHSystem`: System to simulate non-reflecting boundary conditions [@Lastiwka:2009]

The code presented here can be found on [GitHub](https://github.com/trixi-framework/TrixiParticles.jl),  along with a detailed [manual](https://trixi-framework.github.io/TrixiParticles.jl/stable/) explaining how to use the package. Additionally, we provide tutorials explaining how to setup a simulation for fluids, structure mechanics or a fluid-structure interaction. All implemented examples for testing are located in the [`examples/`](https://github.com/trixi-framework/TrixiParticles.jl/tree/main/examples) folder, thus, the results can be easily reproduced.

### Feature highlights

As a young project that aims to be the Julia code for particle-based simulation we expect to integrate other methods such as incompressible SPH (ISPH) [@Liu:2022] or heat conduction with SPH [@Biriukov:2018] and couple them among each other and even with mesh-based methods.

TODO: Instead of listing it in the paper, it might be better to link the [feature list](https://trixi-framework.github.io/TrixiParticles.jl/stable/#Features)
So far, the following feature highlights have been implemented:

* Weakly compressible SPH with density diffusion [@Antuono:2010]

* Entropically damped artificial compressibility (EDAC) [@Ramachandran:2019]

* Total Lagrangian SPH (TLSPH) and fluid-structure interaction [@O_Connor:2021]

* Discrete element method (DEM) [@Bicanic:2004], [@Cundall:1979]

* Transport-velocity formulation (TVF) [@Adami:2013]

* Intra-particle-force surface tension [@Akinci:2013]

* Non-reflecting (open) boundaries [@Lastiwka:2009]

* Efficient grid neighborhood search [@Ihmsen:2010]

* GPU support

Figure \autoref{fig:falling_sphere} illustrates an example of our simulation results. In this example, an elastic sphere modeled with TLSPH falls into a tank filled with water.

![Elastic sphere falling into a tank filled with water. \label{fig:falling_sphere}](falling_sphere_combined_nonstick_4k_178.png){width=50%}

The current state allows also to validate our simulation and produce quantitative results with a post-process callback. \autoref{fig:beam_y_deflection} shows simulation results of TrixiParticles.jl (on the left) and [@O_Connor:2021] (on the right) compared against a reference value of [@Turek:2007]. The curves show the y-deflection of the tip of a beam oscillating under its own weight. The results obtained with TrixiParticles.jl match perfectly those of [@O_Connor:2021].

![Comparison of TrixiParticles.jl and  [@O_Connor:2021] against [@Turek:2007]: Tip y-deflection of an oscillating beam with different resolutions, where $t_s$ is the thickness of the beam and $dp$ is the particle spacing. \label{fig:beam_y_deflection}](oscillating_beam.png){width=100%}

# Acknowledgements

TODO: What else?
The project has benefited from funding from [hereon](https://www.hereon.de/) and [HiRSE](https://www.helmholtz-hirse.de/).

# References
