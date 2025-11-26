# # Modifying or extending components of TrixiParticles.jl within a simulation file

# In this tutorial, we will show how to replace components
# of TrixiParticles.jl by custom implementations from within a simulation file,
# without ever cloning the repository.

# First, we import TrixiParticles.jl and
# [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl), which we will
# use at the very end for the time integration.
using TrixiParticles
using OrdinaryDiffEq

# ## Load a simulation file

# Instead of setting up the simulation from scratch, we run a pre-defined
# simulation file that implements a 2D dam break problem with
# Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH).
# This tutorial works with any other simulation file as well, as long as
# it contains an assignment `smoothing_kernel = ...`.
# By using [`trixi_include`](@ref), we can overwrite variables defined
# in the simulation file.
# Here, we pass `sol=nothing` to overwrite the line that defines the variable `sol`,
# which is the line that runs the actual simulation.
# This way, we can import all variables and definitions from the simulation file
# without running the simulation right away.
# In this tutorial, we only need the definition of the particle spacing,
# but this approach works for any variable defined in the simulation file
# and is used in many pre-defined example simulation files.
# See [the tutorial on setting up a simulation](@ref tut_setup) for details on the
# general structure of a simulation file.
trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
              sol=nothing)

# ## Replacing components with custom implementations

# In this tutorial, we want to replace the smoothing kernel used in the simulation
# by a custom implementation.
# A good starting point is to check out the available implementations in
# TrixiParticles.jl, then copy the relevant functions to the simulation file
# and modify them as needed.

# ### Custom smoothing kernel

# To implement a custom smoothing kernel, we define a struct extending
# `TrixiParticles.AbstractSmoothingKernel`.
# This abstract struct has a type parameter for the number of dimensions,
# which we set to 2 in this case.
struct MyGaussianKernel <: TrixiParticles.AbstractSmoothingKernel{2} end
# This kernel is going to be an implementation of the Gaussian kernel with
# a cutoff for compact support, which reads
# ```math
# W(r, h) =
# \begin{cases}
# \frac{1}{\pi h^2} \exp(-(r/h)^2) & \text{for } r < 2h\\
# 0 & \text{for } r \geq 2h.
# \end{cases}
# ```
# Note that the same kernel in a more optimized version and with a cutoff at ``3``
# is already implemented in TrixiParticles.jl as [`GaussianKernel`](@ref).

# By looking at the implementation of existing kernels in TrixiParticles.jl,
# we can see that a kernel implementation requires three functions.
# `TrixiParticles.kernel`, which is the kernel function itself,
# `TrixiParticles.kernel_deriv`, which is the derivative of the kernel function,
# and `TrixiParticles.compact_support`, which defines the compact support of the
# kernel in relation to the smoothing length.
# The latter is relevant for determining the search radius of the neighborhood search.
function TrixiParticles.kernel(kernel::MyGaussianKernel, r, h)
    q = r / h

    if q < 2
        return 1 / (pi * h^2) * exp(-q^2)
    end

    return 0.0
end

function TrixiParticles.kernel_deriv(kernel::MyGaussianKernel, r, h)
    q = r / h

    if q < 2
        return 1 / (pi * h^2) * (-2 * q) * exp(-q^2) / h
    end

    return 0.0
end

TrixiParticles.compact_support(::MyGaussianKernel, h) = 2 * h

# For this kernel, we use a smoothing length of 1.2 times the particle spacing,
# which yields a similar kernel shape as the commonly used Wendland C2 kernel
# with a smoothing length of 1.59 times the particle spacing.
smoothing_length = 1.2 * fluid_particle_spacing
smoothing_length_wendland = 1.59 * fluid_particle_spacing
nothing # hide
# We can compare these kernels in a plot.
using Plots
plot(r -> TrixiParticles.kernel(WendlandC2Kernel{2}(), abs(r), smoothing_length_wendland),
     -3 * fluid_particle_spacing, 3 * fluid_particle_spacing,
     label="WendlandC2Kernel", xlabel="r")
plot!(r -> TrixiParticles.kernel(MyGaussianKernel(), abs(r), smoothing_length),
      label="MyGaussianKernel")
plot!(dpi=200) # hide
savefig("tut_custom_kernel_plot2.png"); # hide
# ![plot](tut_custom_kernel_plot2.png)

# This is all we need to use our custom kernel implementation in a simulation.
# We only need to replace the definition above by
smoothing_kernel = MyGaussianKernel()
nothing # hide
# and run the simulation file again.

# In order to use our kernel in a pre-defined example file, we can use the function
# [`trixi_include`](@ref) to replace the definition of the variable `smoothing_kernel`.
# The following will run the example simulation
# `examples/fluid/dam_break_2d.jl` with our custom kernel and the corresponding
# smoothing length.
# ```@cast @__NAME__; width=100, height=50, delay=0, loop=true, loop_delay=5
# trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "dam_break_2d.jl"),
#               smoothing_kernel=MyGaussianKernel(),
#               smoothing_length=smoothing_length);
# ```
trixi_include(joinpath(examples_dir(), "fluid", "dam_break_2d.jl"), #!md
              smoothing_kernel=MyGaussianKernel(), #!md
              smoothing_length=smoothing_length) #!md

# See [Visualization](@ref) for how to visualize the final solution.
# For the simplest visualization, we can use [Plots.jl](https://docs.juliaplots.org/stable/):
using Plots
plot(sol, ylims=(-Inf, 1))
plot!(dpi=200) # hide
savefig("tut_custom_kernel_plot.png"); # hide
# ![plot](tut_custom_kernel_plot.png)
