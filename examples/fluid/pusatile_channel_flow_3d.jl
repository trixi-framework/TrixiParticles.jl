using TrixiParticles

particle_count_spanning_pipe = 10

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hagen_poiseuille_flow_3d.jl"),
              tspan=(0.0, 1.0), sol=nothing,
              particle_count_spanning_pipe=particle_count_spanning_pipe)

v_max = pipe_diameter^2 * pressure_drop / (8 * dynamic_viscosity * flow_length)

function dynamic_pressure_drop(pos, t)
    return pressure_drop + (flow_length - pos[1]) / flow_length * pressure_drop * cos(t)
end

trixi_include(@__MODULE__, joinpath(examples_dir(), "fluid", "hagen_poiseuille_flow_3d.jl"),
              tspan=(0.0, 12.0), particle_count_spanning_pipe=particle_count_spanning_pipe,
              v_max=v_max,
              reference_pressure_in=dynamic_pressure_drop,
              reference_pressure_out=dynamic_pressure_drop)
