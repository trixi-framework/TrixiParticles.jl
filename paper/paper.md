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

Numerical simulations such as computational fluid dynamics, structural mechanics, thermodynamics
or magnetohydrodynamics are often simulated using mesh-based methods. A wide range of
mature and powerful software is available for these simulations. However, in certain applications,
these methods reach their limits or are not designed to solve these problems efficiently.
This is where particle-based methods offer an alternative.

The main applications of particle-based methods such as Smoothed Particle Hydrodynamics (SPH),
include hydrodynamics for the simulation of fluids with free surfaces, multiphase flows with
complex interfaces, solid mechanics and also the simulation of heat conduction problems.

The mesh-free formalism of the methods allows for easy preprocessing, making them particularly
suitable for simulating complex geometries. This also facilitates the coupling of different single
physics into a multiphysics system.

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

Feature highlights:

- Weakly compressible SPH with density diffusion [@Antuono:2010]
- Entropically damped artificial compressibility (EDAC) [@Ramachandran:2019]
- Transport-velocity formulation (TVF) [@Adami:2013]
- Intra-particle-force surface tension
- Non-reflecting (open) boundaries [@Lastiwka2009]
- Total Lagrangian SPH and fluid-structure interaction [@O_Connor:2021]
- GPU support

Figure \autoref{fig:falling_sphere} illustrates an example of our simulation results.
In this example, an elastic sphere modeled with Total Lagrangian SPH falls into a tank filled with water.

![Elastic sphere falling into a tank filled with water.\label{fig:falling_sphere}](falling_sphere_combined_nonstick_4k_178.png){width=50%}

# Citations

# Acknowledgements
The project has benefited from funding from [hereon](https://www.hereon.de/) and [HiRSE](https://www.helmholtz-hirse.de/).

# References
