using TrixiParticles
using OrdinaryDiffEq
using Plots

tfv = true
output_directory = "out" #tfv ? "out_tvf" : "out_pst"

# ==========================================================================================
# ==== Resolution
const wall_distance = 0.001
const flow_length = 0.004

particle_spacing = wall_distance / 50

# Make sure that the kernel support of fluid particles at a boundary is always fully sampled
boundary_layers = 4

# Make sure that the kernel support of fluid particles at an open boundary is always
# fully sampled.
open_boundary_layers = 10

# ==========================================================================================
# ==== Experiment Setup
tspan = (0.0, 2.0)

domain_size = (flow_length, wall_distance)

open_boundary_size = (open_boundary_layers * particle_spacing, domain_size[2])

fluid_density = 1000.0
reynolds_number = 50
const pressure_drop = 0.1
const dynamic_viscosity = sqrt(fluid_density * wall_distance^3 * pressure_drop /
                               (8 * flow_length * reynolds_number))

v_max = wall_distance^2 * pressure_drop / (8 * dynamic_viscosity * flow_length)

sound_speed = 100 * v_max

flow_direction = (1.0, 0.0)

function poiseuille_velocity(y, t)

    # Base profile (stationary part)
    base_profile = (pressure_drop / (2 * dynamic_viscosity * flow_length)) * y *
                   (y - wall_distance)

    # Transient terms (Fourier series)
    transient_sum = 0.0

    for n in 0:10  # Limit to 10 terms for convergence
        coefficient = (4 * pressure_drop * wall_distance^2) /
                      (dynamic_viscosity * flow_length * π^3 * (2 * n + 1)^3)

        sine_term = sin(π * y * (2 * n + 1) / wall_distance)

        exp_term = exp(-((2 * n + 1)^2 * π^2 * dynamic_viscosity * t) /
                       (fluid_density * wall_distance^2))

        transient_sum += coefficient * sine_term * exp_term
    end

    # Total velocity
    v_x = base_profile + transient_sum

    return v_x
end

# For the inflow velocity
function velocity_profile(pos, t)
    y = pos[2]  # y-coordinate
    v_x = poiseuille_velocity(y, t)

    return (-v_x, 0.0)  # (v_x, v_y)
end

pipe = RectangularTank(particle_spacing, domain_size, domain_size, fluid_density,
                       velocity=(pos) -> velocity_profile(pos, 0),
                       pressure=(pos) -> 0.2 + (0.1 - 0.2) * (pos[1] / flow_length),
                       n_layers=boundary_layers, faces=(false, false, true, true))

inlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                        fluid_density, n_layers=boundary_layers,
                        velocity=(pos) -> velocity_profile(pos, 0), pressure=0.2,
                        min_coordinates=(-open_boundary_size[1], 0.0),
                        faces=(false, false, true, true))

outlet = RectangularTank(particle_spacing, open_boundary_size, open_boundary_size,
                         fluid_density, n_layers=boundary_layers,
                         velocity=(pos) -> velocity_profile(pos, 0), pressure=0.1,
                         min_coordinates=(pipe.fluid_size[1], 0.0),
                         faces=(false, false, true, true))

n_buffer_particles = 10 * pipe.n_particles_per_dimension[2]^2

# ==========================================================================================
# ==== Fluid
smoothing_length = 2 * particle_spacing
smoothing_kernel = WendlandC2Kernel{2}()

fluid_density_calculator = ContinuityDensity()

kinematic_viscosity = dynamic_viscosity / fluid_density

viscosity = ViscosityAdami(nu=kinematic_viscosity)

state_equation = StateEquationCole(; sound_speed, reference_density=fluid_density,
                                   exponent=1)

if tfv
    background_pressure = 7 * fluid_density * v_max^2
    transport_velocity = TransportVelocityAdami(; background_pressure)
else
    transport_velocity = nothing
end

fluid_system = WeaklyCompressibleSPHSystem(pipe.fluid, fluid_density_calculator,
                                           state_equation, smoothing_kernel,
                                           buffer_size=n_buffer_particles,
                                           shifting_technique=transport_velocity,
                                           density_diffusion=DensityDiffusionMolteniColagrossi(delta=0.1),
                                           smoothing_length, viscosity=viscosity)

# state_equation = nothing

# fluid_system = EntropicallyDampedSPHSystem(pipe.fluid, smoothing_kernel, smoothing_length,
#                                            sound_speed, viscosity=viscosity,
#                                            density_calculator=fluid_density_calculator,
#                                            shifting_technique=transport_velocity,
#                                            buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Open Boundary
boundary_type_in = BidirectionalFlow()
plane_in = ([0.0, 0.0], [0.0, pipe.fluid_size[2]])
inflow = BoundaryZone(; plane=plane_in, plane_normal=flow_direction, open_boundary_layers,
                      density=fluid_density, particle_spacing,
                      #   reference_velocity=velocity_profile,
                      #   rest_pressure=0.2,
                      reference_pressure=0.2,
                      initial_condition=inlet.fluid, boundary_type=boundary_type_in)

boundary_type_out = BidirectionalFlow()
plane_out = ([pipe.fluid_size[1], 0.0], [pipe.fluid_size[1], pipe.fluid_size[2]])
outflow = BoundaryZone(; plane=plane_out, plane_normal=(.-(flow_direction)),
                       reference_pressure=0.1,
                       #   reference_velocity=velocity_profile,
                       open_boundary_layers, density=fluid_density, particle_spacing,
                       initial_condition=outlet.fluid, boundary_type=boundary_type_out)

open_boundary = OpenBoundarySystem(inflow, outflow; fluid_system,
                                   boundary_model=BoundaryModelDynamicalPressureZhang(),
                                   buffer_size=n_buffer_particles)

# ==========================================================================================
# ==== Boundary
wall = union(pipe.boundary, inlet.boundary, outlet.boundary)

boundary_model = BoundaryModelDummyParticles(wall.density, wall.mass,
                                             AdamiPressureExtrapolation(),
                                             state_equation=state_equation,
                                             viscosity=viscosity,
                                             smoothing_kernel, smoothing_length)

boundary_system = WallBoundarySystem(wall, boundary_model)

# ==========================================================================================
# ==== Simulation
min_corner = minimum(inlet.boundary.coordinates .- 2 * particle_spacing, dims=2)
max_corner = maximum(outlet.boundary.coordinates .+ 2 * particle_spacing, dims=2)

nhs = GridNeighborhoodSearch{2}(; cell_list=FullGridCellList(; min_corner, max_corner),
                                update_strategy=ParallelUpdate())

semi = Semidiscretization(fluid_system, open_boundary,
                          boundary_system, neighborhood_search=nhs,
                          parallelization_backend=PolyesterBackend())

ode = semidiscretize(semi, tspan)

info_callback = InfoCallback(interval=100)
saving_callback = SolutionSavingCallback(dt=0.02, prefix="",
                                         output_directory=output_directory)

v_x_interpolated(system, dv_ode, du_ode, v_ode, u_ode, semi, t) = nothing
function v_x_interpolated(system::TrixiParticles.AbstractFluidSystem, dv_ode, du_ode, v_ode, u_ode,
                          semi, t)
    start_point = [flow_length / 2, 0.0]
    end_point = [flow_length / 2, wall_distance]

    values = interpolate_line(start_point, end_point, 100, semi, system, v_ode, u_ode;
                              cut_off_bnd=true,
                              clip_negative_pressure=false)

    return values.velocity[1, :]
end

pp_callback = PostprocessCallback(; dt=0.02,
                                  output_directory=output_directory,
                                  v_x=v_x_interpolated, filename="result",
                                  write_csv=true, write_file_interval=1)

extra_callback = nothing

callbacks = CallbackSet(info_callback, saving_callback, UpdateCallback(),
                        pp_callback, extra_callback)

sol = solve(ode, RDPK3SpFSAL35(),
            abstol=1e-6, # Default abstol is 1e-6 (may need to be tuned to prevent boundary penetration)
            reltol=1e-4, # Default reltol is 1e-3 (may need to be tuned to prevent boundary penetration)
            dtmax=1e-2, # Limit stepsize to prevent crashing
            save_everystep=false, callback=callbacks);

data = TrixiParticles.CSV.read(joinpath(output_directory, "result.csv"),
                               TrixiParticles.DataFrame)

times = data[!, "time"]

target_time = 0.1
idx = argmin(abs.(times .- target_time))  # Nächster verfügbarer Zeitpunkt

# Parsen der String-Arrays zu echten Arrays
v_x_arrays = [eval(Meta.parse(str)) for str in data[!, "v_x_fluid_1"]]

p = scatter(range(0, wall_distance, length=50), v_x_arrays[idx][1:2:100],
            label="TrixiP (0.1s)",
            linewidth=3,
            color=:red, markersize=5, opacity=0.4)
plot!(p, (y) -> -poiseuille_velocity(y, target_time), xlims=(0, wall_distance),
      ylims=(-0.002, 0.014), label="analytical", linewidth=3, linestyle=:dash, color=:black)

target_time = 0.3
idx = argmin(abs.(times .- target_time))  # Nächster verfügbarer Zeitpunkt
scatter!(p, range(0, wall_distance, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (0.3s)", linewidth=3,
         color=:red, markersize=5, opacity=0.4)
plot!(p, (y) -> -poiseuille_velocity(y, target_time), xlims=(0, wall_distance),
      ylims=(-0.002, 0.014), label="analytical", linewidth=3, linestyle=:dash, color=:black)

target_time = 0.6
idx = argmin(abs.(times .- target_time))  # Nächster verfügbarer Zeitpunkt
scatter!(p, range(0, wall_distance, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (0.6s)", linewidth=3,
         color=:red, markersize=5, opacity=0.4)
plot!(p, (y) -> -poiseuille_velocity(y, target_time), xlims=(0, wall_distance),
      ylims=(-0.002, 0.014), label="analytical", linewidth=3, linestyle=:dash, color=:black)

target_time = 0.9
idx = argmin(abs.(times .- target_time))  # Nächster verfügbarer Zeitpunkt
scatter!(p, range(0, wall_distance, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (0.9s)", linewidth=3,
         color=:red, markersize=5, opacity=0.4)
plot!(p, (y) -> -poiseuille_velocity(y, target_time), xlims=(0, wall_distance),
      ylims=(-0.002, 0.014), label="analytical", linewidth=3, linestyle=:dash, color=:black)

target_time = tspan[2]
idx = argmin(abs.(times .- target_time))  # Nächster verfügbarer Zeitpunkt
scatter!(p, range(0, wall_distance, length=50), v_x_arrays[idx][1:2:100],
         label="TrixiP (∞)", linewidth=3,
         color=:red, markersize=5, opacity=0.4)
plot!(p, (y) -> -poiseuille_velocity(y, target_time), xlims=(0, wall_distance),
      legendposition=:outerright, size=(750, 400),
      ylims=(-0.002, 0.014), label="analytical", linewidth=3, linestyle=:dash, color=:black)

rmsep = Float64[]
for t_target in (0.1, 0.3, 0.6, 0.9, 2.0)
    idx = argmin(abs.(times .- t_target))  # Nächster verfügbarer Zeitpunkt

    range_ = 10:90
    N = length(range_)
    positions = range(0, wall_distance, length=100)
    res = sum(range_, init=0) do i
        v_x = v_x_arrays[idx][i]

        v_analyitcal = -poiseuille_velocity(positions[i], t_target)

        v_analyitcal < sqrt(eps()) && return 0.0

        rel_err = (v_analyitcal - v_x) / v_analyitcal

        return rel_err^2 / N
    end

    push!(rmsep, sqrt(res) * 100)
end

rmsep
