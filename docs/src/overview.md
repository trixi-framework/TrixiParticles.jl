# Overview
The following page gives a rough overview of important parts of the code.

## Program flow

To initiate a simulation, the goal is to solve an ordinary differential equation, for example, by employing the time integration schemes provided by OrdinaryDiffEq.jl. These schemes are then utilized to integrate du/dt and dv/dt, where 'u' represents the particles' position and 'v' signifies their properties, such as velocity and density. During a single time step or an intermediate step of the time integration scheme, the functions `drift!` and `kick!` are invoked, followed by the functions depicted in this diagram (with key parts highlighted in orange/yellow).
![Main Program Flow](https://github.com/trixi-framework/TrixiParticles.jl/assets/10238714/d7c6eedd-7173-4879-b62e-3e6d4bc5091f)


## Structure
What we refer to as schemes are various models such as Weakly Compressible Spherical Particle Hydrodynamics (WCSPH) or Total Lagrangian Spherical Particle Hydrodynamics (TLSPH). These schemes are categorized based on the applicable physical regimes, namely fluid, solid, gas, and others. Each scheme comprises at least two files: a **system.jl** file and an **rhs.jl** file. The **system.jl** file provides most of the other routines, particularly those for allocation and the main update routines, excluding system interactions. The interactions within the system are handled in the rhs.jl file.
