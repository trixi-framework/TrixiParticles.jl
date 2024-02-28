# Results are compared to the results in:
#
# P.N. Sun, D. Le Touzé, A.-M. Zhang.
# "Study of a complex fluid-structure dam-breaking benchmark problem using a multi-phase SPH method with APR".
# In: Engineering Analysis with Boundary Elements 104 (2019), pages 240-258.
# https://doi.org/10.1016/j.enganabound.2019.03.033
# and
# Turek S , Hron J.
# "Proposal for numerical benchmarking of fluid-structure interaction between an elastic object and laminar incompressible flow."
# In: Fluid-structure interaction. Springer; 2006. p. 371–85 .
# https://doi.org/10.1007/3-540-34596-5_15

using TrixiParticles
using OrdinaryDiffEq

tspan = (0, 10)

# for the brave add 35
resolution = [9, 21]
for res in resolution
    trixi_include(@__MODULE__,
                  joinpath(examples_dir(), "solid", "oscillating_beam_2d.jl"),
                  n_particles_y=res, sol=nothing, tspan=tspan,
                  penalty_force=PenaltyForceGanzenmueller(alpha=0.01))

    pp_callback = PostprocessCallback(; deflection_x, deflection_y, dt=0.01,
                                      output_directory="out",
                                      filename="validation_run_oscillating_beam_2d_" *
                                               string(res), write_csv=false,
                                      write_file_interval=0)
    info_callback = InfoCallback(interval=2500)
    saving_callback = SolutionSavingCallback(dt=0.5, prefix="validation_" * string(res))

    callbacks = CallbackSet(info_callback, saving_callback, pp_callback)

    sol = solve(ode, RDPK3SpFSAL49(), abstol=1e-8, reltol=1e-6, dtmax=1e-3,
                save_everystep=false, callback=callbacks)
end
