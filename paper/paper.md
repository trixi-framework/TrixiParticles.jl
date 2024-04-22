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
date: 19 April 2024 # TODO
bibliography: paper.bib
---

# Summary

TrixiParticles.jl, part of the Trixi Project [@schlottkelakemper2020trixi], is an innovative Julia-based open-source framework designed for particle-based multiphysics simulations.
It aims to overcome the limitations of traditional mesh-based methods in handling complex geometries and specialized applications, such as computational fluid dynamics and structural mechanics,
by offering a versatile platform for Smoothed Particle Hydrodynamics (SPH) and the Discrete Element Method (DEM), among others.
Unique in its approach, TrixiParticles.jl facilitates the easy addition of new particle systems and their interactions, enhancing research flexibility.
This is further complemented by its user-friendly design, which allows simulations to be configured directly with Julia code, eliminating the need for parameter files.
This feature not only simplifies the integration of custom functionalities but also promotes rapid prototyping, establishing TrixiParticles.jl as a robust and adaptable tool for advancing multiphysics simulations in various scientific and engineering fields.

Here, we give a brief overview of the software TrixiParticles.jl, starting with a description of the scientific background before going on to describe the functionality and usefulness in more detail.
Finally, some of the results obtained and some of the features that have been implemented so far are briefly presented.

# Statement of need

Numerical simulations, such as computational fluid dynamics, structural mechanics, thermodynamics, or magnetohydrodynamics, are often conducted using mesh-based methods supported by mature software tools like ANSYS Fluent, COMSOL Multiphysics, and OpenFOAM. However, these methods face limitations for complex geometries or when integrating different physics disciplines into a single multiphysics system.

The mesh-free formalism of the methods allows for easy preprocessing, making them particularly
suitable for simulating complex geometries. This also facilitates the coupling of different single
physics into a multiphysics system.

There are several open-source software projects specialized for SPH methods, including [@DualSPHysics], [@SplisHSPlasH] and [@SPHinXsys],
written in C++, and  [@PySPH], written in Python. These frameworks emphasize performance and are designed for a variety of SPH schemes.

TrixiParticles.jl provides support for developing or testing new SPH methods and also for simulating or coupling other particle-based methods such as DEM.
Another aspect is that TrixiParticles.jl is not configured at runtime via a parameter file, as it is the case with other numerical simulation codes.
Instead, each simulation is configured and set up by pure Julia code.
This makes it easy for users to add custom functionality even without touching the source code.
The latter is further enhanced by its seamless support for GPU acceleration, integrating advanced computational capabilities without necessitating separate codebases.

# Scientific background

In TrixiParticles.jl, particles of a single particle-based method are grouped into a so called \emph{system}.
The interaction between two particles is defined by the types of their systems. This approach makes it easy to add new methods and different physics.

![Particles of two different systems in a simulation domain. \label{fig:systems}](systems.png){width=40%}

To illustrate this, \autoref{fig:systems} shows particles in a simulation domain. The black particles belong to system $\mathcal{S}_1$ and the gray particles belong to system $\mathcal{S}_2$.
In general, the force $f_a$ experienced by a particle $a$ is calculated as following
$$ f_a = \sum_{b \in \mathcal{S}_1} f_{ab}^{\mathcal{S}_1} + \sum_{b \in \mathcal{S}_2} f_{ab}^{\mathcal{S}_2} + \dots + \sum_{b\in \mathcal{S}_n}f_{ab}^{\mathcal{S}_n}, $$
where the interaction force $f_{ab}^{\mathcal{S}_i}$ that particle $a$ experiences due to particle $b$ depends on the system type of particle $a$ and the type of the system $\mathcal{S}^i$ of particle $b$ is the set of neighboring particles of the corresponding system denoted by the superscript and $f_{ab}$ is the interaction force of particle $a$ and neighbor particle $b$.
For computational efficiency, only particles with a distance within a system-dependent search radius interact with each other.

For example, the SPH method determines the force between two SPH particles according to [@Monaghan:2005] as
$$ f_{ab} = -m_a m_b \left( \frac{p_a}{\rho_a^2} + \frac{p_b}{\rho_b^2} \right) \nabla_a W_{ab} + \Pi_{ab},$$
where $m_a$, $m_b$, $\rho_a$, $\rho_b$, $p_a$, $p_b$ are the mass, density and pressure of particles $a$ and $b$, respectively. The last term $\Pi_{ab}$ includes dissipative terms such as artificial viscosity [@Monaghan:2005] and is scheme-specific. The weighting function $W_{ab}$, also called kernel-function, depends on the relative distance between particles $a$ and $b$.

# Functionality

\autoref{fig:structure} depicts the basic building blocks of TrixiParticles.jl. A simulation essentially consists of spatial discretization (left block) and time integration (center block). For the latter, the Julia package [OrdinaryDiffEq.jl](https://docs.sciml.ai/DiffEqDocs/latest/) is used. The callbacks (right block) are used to communicate with the time integration interface during the simulation.

The object `Semidiscretization` couples the systems of a simulation and also creates the corresponding neighborhood searches for each system. The resulting ODE problem is then fed into the time integrator and is solved by an appropriate numerical time integration scheme.

![Main building blocks of TrixiParticles.jl \label{fig:structure}](structure.png){width=100%}

The code presented here can be found on [GitHub](https://github.com/trixi-framework/TrixiParticles.jl),  along with a detailed [manual](https://trixi-framework.github.io/TrixiParticles.jl/stable/) explaining how to use the package. Additionally, we provide tutorials explaining how to set up a simulation for fluids, structure mechanics or a fluid-structure interaction.
A collection of predefined simulation files can be found in the `examples` directory.

### Feature highlights

So far, the following feature highlights have been implemented:

* *Fluid Systems*
    + `WeaklyCmpressibleSPHSystem` (WCSPH): Standard SPH method originally developed by [@Monaghan:1977] to simulate astrophysics applications.
    + `EntropicallyDampedSPHSystem` (EDAC): As opposed to the WCSPH scheme, which uses an equation of state, this scheme uses a pressure evolution equation to calculate the pressure which is derived by [@Clausen:2013] and adapted to SPH by [@Ramachandran:2019].

* *Structure Systems*
    + `TotalLagrangianSPHSystem` (TLSPH): System to simulate elastic structure where the interaction is defined with stress tensors [@O_Connor:2021].
    + `DEMSystem`: Discrete element system [@Bicanic:2004], [@Cundall:1979].

* *Boundary Systems*
    + `BoundarySPHSystem` with several boundary models where each model follows a different interaction rule.
    + `OpenBoundarySPHSystem`: System to simulate non-reflecting (open) boundary conditions [@Lastiwka:2009]

* *Correction methods and models*
  + Density diffusion [@Antuono:2010]
  + Transport-velocity formulation (TVF) [@Adami:2013]
  + Intra-particle-force surface tension [@Akinci:2013]
  + Kernel corrections TODO: refs

* *TODO: category naming?*
  + Efficient grid neighborhood search
  + GPU support

\autoref{fig:falling_sphere} illustrates an example of our simulation results. In this example, an elastic sphere modeled with TLSPH falls into a tank filled with water, modeled by WCSPH.

![Elastic sphere falling into a tank filled with water. \label{fig:falling_sphere}](falling_sphere_combined_nonstick_4k_178.png){width=50%}

The current state allows also to validate our simulation and produce quantitative results with a post-process callback. \autoref{fig:beam_y_deflection} shows simulation results of TrixiParticles.jl (on the left) and [@O_Connor:2021] (on the right) compared against a reference value of [@Turek:2007].
The curves show the y-deflection of the tip of a beam oscillating under its own weight.
The results obtained with TrixiParticles.jl perfectly match those of [@O_Connor:2021].

![Comparison of TrixiParticles.jl and  [@O_Connor:2021] against [@Turek:2007]: Tip y-deflection of an oscillating beam with different resolutions, where $t_s$ is the thickness of the beam and $dp$ is the particle spacing. \label{fig:beam_y_deflection}](oscillating_beam.png){width=100%}

As a young project that aims to be the Julia code for particle-based simulation we expect to integrate other methods such as incompressible SPH (ISPH) [@Liu:2022] or heat conduction with SPH [@Biriukov:2018] and couple them among each other and even with mesh-based methods.

# Acknowledgements

TODO: What else?
The project has benefited from funding from [hereon](https://www.hereon.de/) and [HiRSE](https://www.helmholtz-hirse.de/).

# References