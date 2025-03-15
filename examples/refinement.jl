using TrixiParticles
using Plots

trixi_include(joinpath(examples_dir(), "refinement_minimal_example.jl"), sol=nothing)

TrixiParticles.foreach_system(semi) do system
    TrixiParticles.resize_refinement!(system)
end

v_ode, u_ode = ode.u0.x

x_lims = (-2particle_spacing, initial_fluid_size[1])
y_lims = (-particle_spacing, initial_fluid_size[2])

smoothing_lengths_fluid = [TrixiParticles.smoothing_length(fluid_system, i)
                           for i in TrixiParticles.eachparticle(fluid_system)]
smoothing_lengths_boundary = [TrixiParticles.smoothing_length(boundary_system, i)
                              for i in TrixiParticles.eachparticle(boundary_system)]

p1 = plot(fluid_system.initial_condition, size=(800, 800), xlims=x_lims, ylims=y_lims,
          label=nothing, zcolor=smoothing_lengths_fluid, color=:redsblues)
plot!(p1, boundary_system.initial_condition, zcolor=smoothing_lengths_boundary,
      label=nothing, xlims=x_lims, ylims=y_lims, size=(800, 800), color=:redsblues)

TrixiParticles.check_refinement_criteria!(semi, v_ode, u_ode)

smoothing_lengths_fluid = [TrixiParticles.smoothing_length(fluid_system, i)
                           for i in TrixiParticles.eachparticle(fluid_system)]
smoothing_lengths_boundary = [TrixiParticles.smoothing_length(boundary_system, i)
                              for i in TrixiParticles.eachparticle(boundary_system)]
p2 = plot(fluid_system.initial_condition, size=(800, 800), xlims=x_lims, ylims=y_lims,
          label=nothing, zcolor=smoothing_lengths_fluid, color=:redsblues)
plot!(p2, boundary_system.initial_condition, zcolor=smoothing_lengths_boundary,
      label=nothing, xlims=x_lims, ylims=y_lims, size=(800, 800), color=:redsblues)

TrixiParticles.update_particle_spacing(semi, v_ode, u_ode)

smoothing_lengths_fluid = [TrixiParticles.smoothing_length(fluid_system, i)
                           for i in TrixiParticles.eachparticle(fluid_system)]
smoothing_lengths_boundary = [TrixiParticles.smoothing_length(boundary_system, i)
                              for i in TrixiParticles.eachparticle(boundary_system)]
p3 = plot(fluid_system.initial_condition, size=(800, 800), xlims=x_lims, ylims=y_lims,
          label=nothing, zcolor=smoothing_lengths_fluid, color=:redsblues)
plot!(p3, boundary_system.initial_condition, zcolor=smoothing_lengths_boundary,
      label=nothing, xlims=x_lims, ylims=y_lims, size=(800, 800), color=:redsblues)

TrixiParticles.split_particles!(semi, v_ode, u_ode, copy(v_ode), copy(u_ode))

smoothing_lengths_fluid = [TrixiParticles.smoothing_length(fluid_system, i)
                           for i in TrixiParticles.eachparticle(fluid_system)]
smoothing_lengths_boundary = [TrixiParticles.smoothing_length(boundary_system, i)
                              for i in TrixiParticles.eachparticle(boundary_system)]
p4 = plot(v_ode, u_ode, semi, size=(800, 800), xlims=x_lims, ylims=y_lims,
          label=nothing;
          particle_spacings=[particle_spacing / 10, particle_spacing / spacing_ratio])

TrixiParticles.merge_particles!(semi, v_ode, u_ode, copy(v_ode), copy(u_ode))
p5 = plot(v_ode, u_ode, semi, size=(800, 800), xlims=x_lims, ylims=y_lims,
          label=nothing;
          particle_spacings=[particle_spacing / 10, particle_spacing / spacing_ratio])

TrixiParticles.upate_smoothing_lengths!(semi, u_ode)
smoothing_lengths_fluid = [TrixiParticles.smoothing_length(fluid_system, i)
                           for i in TrixiParticles.eachparticle(fluid_system)]
smoothing_lengths_boundary = [TrixiParticles.smoothing_length(boundary_system, i)
                              for i in TrixiParticles.eachparticle(boundary_system)]
p6 = plot(v_ode, u_ode, semi, size=(800, 800), xlims=x_lims, ylims=y_lims,
          label=nothing, zcolor=smoothing_lengths_fluid, color=:redsblues;
          particle_spacings=[particle_spacing / 10, particle_spacing / spacing_ratio])
