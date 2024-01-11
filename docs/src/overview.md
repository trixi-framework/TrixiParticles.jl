# Overview
The following page will give a rough overview of important parts of the code.

## Program flow
By starting a simulation one tries to solve a ordinary differential equation e.g. by using the time integration schemes provied by OrdinaryDiffEq.jl. These can than be used to integrate du/dt and dv/dt.
With u being the position of the particles and v the properties i.e. velocity, density. 
During one time step or intermediate step of the time integration scheme the functions drift! and kick! are called followed by the functions shown in this diagram (important parts are highlighted in orange/yellow)
![Main Program Flow](https://github.com/svchb/TrixiParticles.jl/assets/10238714/c11bce52-7179-481e-96ad-e7ba146c6860)


## Structure
What we call schemes are different models like weakly compressible spherical particle hydrodynamics (WCSPH) or total lagrangragian spherical particle hydrodynamics (TLSPH). These schemes are divided into the applicable physical regime
i.e. fluid, solid, gas and so on. Each scheme is made up of at least a system.jl and rhs.jl file. The system.jl file provides most other rountines especially the allocation and main update routines except for the system interactions. The system interactions are located within the rhs.jl file.
