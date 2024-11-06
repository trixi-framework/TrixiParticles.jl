using TrixiParticles

# n_procs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
n_procs = 6
particle_spacing = 0.03

min_corner_global = SVector(-1.1, -1.0, -1.0)
max_corner_global = SVector(1.0, 1.0, 1.0)

filename = "aorta"
file = joinpath("examples", "preprocessing", "data", filename * ".stl")

decomp_geometry = load_geometry(file; n_procs)

# TODO: balancing
n_bbox = length(decomp_geometry.bboxes)
sizes_bbox = last.(decomp_geometry.bboxes)
n_faces_per_bbox = length.(decomp_geometry.face_ids)

grid = TrixiParticles.sample_particles(decomp_geometry, particle_spacing;
                                       winding_number_factor=0.4)

trixi2vtk(grid)

function benchmark2(file, particle_spacing, n_procs)
    decomp_geometry = load_geometry(file; n_procs)
    grid = TrixiParticles.sample_particles(decomp_geometry, particle_spacing;
                                           winding_number_factor=0.4)
    trixi2vtk(grid)
end
