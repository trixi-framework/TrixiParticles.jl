module HydrostaticPlateValidation

using TrixiParticles
using LinearAlgebra: norm
using SparseArrays: sparse, spdiagm

const TP = TrixiParticles

# Match the example's material geometry, including full particle volumes on the shell
# and three fixed columns outside either end. Tests compare this setup with the example.
function plate_system(resolution; plate_size=(1.0, 0.05), structure_density=2700.0,
                      young_modulus=67.5e9, poisson_ratio=0.3)
    resolution >= 3 ||
        throw(ArgumentError("use at least three particles through the plate"))
    length_, thickness = plate_size
    spacing = thickness / (resolution - 1)
    nx = round(Int, length_ / spacing + 1)
    plate = RectangularShape(spacing, (nx, resolution), (0.0, -thickness);
                             density=structure_density, place_on_shell=true)
    left = RectangularShape(spacing, (3, resolution), (-3spacing, -thickness);
                            density=structure_density, place_on_shell=true)
    right = RectangularShape(spacing, (3, resolution), (length_ + spacing, -thickness);
                             density=structure_density, place_on_shell=true)
    fixed = union(left, right)
    return TotalLagrangianSPHSystem(union(fixed, plate);
                                    smoothing_kernel=WendlandC2Kernel{2}(),
                                    smoothing_length=sqrt(2) * spacing,
                                    young_modulus, poisson_ratio,
                                    clamped_particles=1:nparticles(fixed))
end

# Differentiate the same corrected deformation gradients as the TLSPH RHS. The
# Hessian of the stress-free elastic energy is sparse and avoids a dense finite-
# difference Jacobian. Strain contributions from fixed particles must be retained.
function linearized_stiffness(system, semi)
    coordinates = system.initial_coordinates
    count = nparticles(system)
    free_count = system.n_integrated_particles
    volumes = system.mass ./ system.material_density
    rows, columns = Int[], Int[]
    gradients_x, gradients_y = Float64[], Float64[]
    neighborhood_search = TP.get_neighborhood_search(system, semi)
    for particle in 1:count
        correction = TP.tlsph_correction_matrix(system, particle)
        TP.mapreduce_neighbor(+, coordinates, coordinates, neighborhood_search,
                              semi.parallelization_backend, particle;
                              init=zero(correction[:, 1])) do particle, neighbor,
                                                              pos_diff, distance
            distance < eps(system.smoothing_length) && return zero(pos_diff)
            gradient = volumes[neighbor] * correction *
                       TP.smoothing_kernel_grad_unsafe(system, pos_diff, distance, particle)
            append!(rows, (particle, particle))
            append!(columns, (neighbor, particle))
            append!(gradients_x, (gradient[1], -gradient[1]))
            append!(gradients_y, (gradient[2], -gradient[2]))
            return zero(pos_diff)
        end
    end
    gradient_x = sparse(rows, columns, gradients_x, count, count)[:, 1:free_count]
    gradient_y = sparse(rows, columns, gradients_y, count, count)[:, 1:free_count]
    lambda, mu = system.lame_lambda, system.lame_mu
    c11 = spdiagm(volumes .* (lambda + 2mu))
    c12 = spdiagm(volumes .* lambda)
    c33 = spdiagm(volumes .* mu)
    kxx = gradient_x' * c11 * gradient_x + gradient_y' * c33 * gradient_y
    kyy = gradient_y' * c11 * gradient_y + gradient_x' * c33 * gradient_x
    kxy = gradient_x' * c12 * gradient_y + gradient_y' * c33 * gradient_x
    return [kxx kxy; kxy' kyy]
end

function rhs_linearization_residual(stiffness, ode, system)
    free_count = system.n_integrated_particles
    direction = sin.(0.71 .* (1:(2free_count)))
    perturbation = vec(permutedims(reshape(direction, free_count, 2)))
    velocity, coordinates = Array(ode.u0.x[1]), Array(ode.u0.x[2])
    acceleration = zero(velocity)
    parameters = (; semi=ode.p.semi, split_integration_data=nothing)
    step = 1e-8
    TP.kick!(acceleration, velocity, coordinates + step * perturbation, parameters, 0.0)
    plus = copy(acceleration)
    TP.kick!(acceleration, velocity, coordinates - step * perturbation, parameters, 0.0)
    force = -system.mass[1:free_count]' .*
            reshape((plus - acceleration) / (2step), 2, free_count)
    blocked_force = [force[1, :]; force[2, :]]
    return norm(stiffness * direction - blocked_force) / norm(blocked_force)
end

"""
    static_plate(resolution; verify_rhs=true, ...)

Isolate structural spatial error from fluid evolution and time integration. Apply a
uniform nominal hydrostatic traction to the top row (trapezoidal endpoint weights)
and gravity to the material masses. Solve the stress-free linearized TLSPH equilibrium.
The benchmark reference uses its original rounded `0.0026` coefficient; `beam_reference`
also records the exact Euler–Bernoulli coefficient `1/384`. This is a diagnostic of
the existing discretization, not a modified FSI configuration or an accuracy correction.
"""
function static_plate(resolution; verify_rhs=true, plate_size=(1.0, 0.05),
                      structure_density=2700.0, fluid_density=1000.0, water_height=2.0,
                      gravity=9.81, young_modulus=67.5e9, poisson_ratio=0.3)
    system = plate_system(resolution; plate_size, structure_density, young_modulus,
                          poisson_ratio)
    semi = Semidiscretization(system; neighborhood_search=nothing,
                              parallelization_backend=SerialBackend())
    ode = semidiscretize(semi, (0.0, 1.0); reset_threads=false)
    semi = ode.p.semi
    system = only(semi.systems)
    stiffness = linearized_stiffness(system, semi)
    free_count = system.n_integrated_particles
    x, y = eachrow(system.initial_coordinates[:, 1:free_count])
    length_, thickness = plate_size
    spacing = thickness / (resolution - 1)
    pressure = fluid_density * gravity * water_height
    force = zeros(2free_count)
    top = maximum(y)
    for particle in 1:free_count
        force[free_count + particle] = -gravity * system.mass[particle]
        if isapprox(y[particle], top; atol=eps(thickness))
            endpoint = isapprox(x[particle], 0.0; atol=eps(length_)) ||
                       isapprox(x[particle], length_; atol=eps(length_))
            force[free_count + particle] -= pressure * spacing * (endpoint ? 0.5 : 1.0)
        end
    end
    displacement = stiffness \ force
    middle = argmin((x .- length_ / 2) .^ 2 + (y .+ thickness / 2) .^ 2)
    deflection = displacement[free_count + middle]
    rigidity = young_modulus * thickness^3 / (12 * (1 - poisson_ratio^2))
    load = gravity * (fluid_density * water_height + structure_density * thickness)
    reference = -0.0026 * load * length_^4 / rigidity
    beam_reference = -load * length_^4 / (384rigidity)
    residual = norm(stiffness * displacement - force) / norm(force)
    rhs_residual = verify_rhs ? rhs_linearization_residual(stiffness, ode, system) : missing
    return (; resolution, spacing, free_particles=free_count,
            material_volume=sum(system.mass[1:free_count]) / structure_density,
            vertical_load=sum(force[(free_count + 1):end]), deflection, reference,
            beam_reference, relative_bias=deflection / reference - 1,
            equilibrium_residual=residual, rhs_linearization_residual=rhs_residual)
end

end # module
