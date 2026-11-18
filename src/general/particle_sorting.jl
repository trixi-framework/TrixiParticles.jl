# # Sorting particles by neighborhood search cell in every update of the neighborhood search
#
# This is enabled when the neighborhood search sorts points
# (see `PointNeighbors.sorts_points`), which is the case for
# a `SortingGridNeighborhoodSearch`. Such a neighborhood search can only be used
# with the `CombinedNHSHandler`.
#
# Sorted particles can skip the index lookups (cell list queries) in the neighbor loops,
# since all particles of a cell are contiguous in memory.
# Additionally, this allows for contiguous vector loads instead of gathers when reading
# particle data, which leads to much better performance with SIMD on CPUs.
#
# # Integration order and system order
# The integration arrays `v_ode` and `u_ode` are owned by the time integrator
# (usually owned by OrdinaryDiffEq.jl).
# They are never reordered in place, except by the `SortingCallback`.
# We call the order of the particles in these arrays the "integration order".
# All arrays that TrixiParticles.jl owns (like the particle masses in the systems) are
# permanently stored in the current sorted order, which we call the "system order".
# Therefore, indexing the raw integration arrays with a particle index of a system
# is not safe when sorting is enabled.
#
# Instead, `v_ode` and `u_ode` are copied to sorted arrays (`SortedODEArray`s)
# in system order before they are used (see `sorted_ode_arrays`).
# All functions working with integration arrays (`kick!`, callbacks, postprocessing)
# work with these sorted copies. `update_systems_and_nhs` returns them.
# Callbacks that modify the integration arrays copy the sorted arrays back
# to the integration order at the end (see `unsort_ode_arrays!`).
# The computed `dv_ode` in `kick!` is copied back to the integration order as well.
# With sorting, `wrap_v` and `wrap_u` throw an error for arrays that are not
# `SortedODEArray`s to prevent accessing the integration arrays in the wrong order.
# Only `drift!` works directly with the integration arrays, using `original_index`
# explicitly.
#
# Note that this only works with the `CombinedNHSHandler`, where everything is using
# the same search radius and therefore each system has only one NHS/cell list,
# which means there is only a single NHS per system that defines the order of the particles.
# In every update of the neighborhood search (see `update_nhs!`), the cell list of each
# updated system is updated with the coordinates in the previous system order,
# and all system arrays are reordered with its permutation.
#
# # Sorted system arrays
# Systems have to specify which of their arrays have to be reordered when the particles
# are sorted (see `particle_arrays_to_sort`).
# Arrays that are recomputed in every update after sorting (like the pressure of
# the `WeaklyCompressibleSPHSystem`) don't need to be reordered.
#
# A system is sorted exactly when its cell list is updated. Therefore, systems whose
# cell lists are not updated (like fixed walls, see `points_moving`) are only sorted once
# at the beginning of the simulation.
struct ParticleSorting{I, A, B, V}
    # `original_index[system_index][i]` is the index in the integration arrays
    # of the particle with index `i` in the system arrays.
    # `sorted_index[system_index]` is the inverse of `original_index[system_index]`.
    original_index :: I
    sorted_index   :: I
    index_buffer   :: I
    # The arrays returned by `particle_arrays_to_sort` for each system (without duplicates)
    # and corresponding buffers to reorder them.
    arrays_to_sort :: A
    array_buffers  :: B
    # Integration arrays in system order (of the type `SortedODEArray`).
    v_ode_sorted  :: V
    u_ode_sorted  :: V
    dv_ode_sorted :: V
    # Buffer to reorder `u_ode_sorted` when particles are sorted.
    # `dv_ode_sorted` is used as buffer to reorder `v_ode_sorted`
    # (see `reorder_sorted_ode_arrays!`).
    u_ode_buffer :: V
end

# Integration arrays in system order.
# When particles are sorted, `wrap_v` and `wrap_u` only accept these arrays.
struct SortedODEArray{T, A} <: AbstractVector{T}
    array::A

    function SortedODEArray(array)
        return new{eltype(array), typeof(array)}(array)
    end
end

Base.parent(array::SortedODEArray) = array.array
@inline Base.size(array::SortedODEArray) = size(parent(array))
@propagate_inbounds Base.getindex(array::SortedODEArray, i::Int) = parent(array)[i]
@propagate_inbounds function Base.setindex!(array::SortedODEArray, value, i::Int)
    parent(array)[i] = value
end

@inline function wrap_array(array::SortedODEArray, range, size)
    return wrap_array(parent(array), range, size)
end

@inline set_zero!(array::SortedODEArray) = set_zero!(parent(array))

transfer2cpu(array::SortedODEArray) = SortedODEArray(transfer2cpu(parent(array)))

# `SortedODEArray`s are already in system order.
@propagate_inbounds function wrap_v(v_ode::SortedODEArray, system, semi)
    return wrap_v_unchecked(v_ode, system, semi)
end

@propagate_inbounds function wrap_u(u_ode::SortedODEArray, system, semi)
    return wrap_u_unchecked(u_ode, system, semi)
end

@inline particle_sorting(semi::Semidiscretization) = semi.particle_sorting

# For other semidiscretization types (like `NamedTuple`s in tests), particles are not sorted.
@inline particle_sorting(semi) = nothing

# `wrap_v` and `wrap_u` must not be used with integration arrays in integration order
# when particles are sorted. See the comments at the top of this file.
@inline check_not_sorted(semi) = check_not_sorted(particle_sorting(semi))
@inline check_not_sorted(::Nothing) = nothing

@noinline function check_not_sorted(::ParticleSorting)
    throw(ArgumentError("integration arrays must be copied to sorted arrays with " *
                        "`sorted_ode_arrays` before they can be wrapped when particles " *
                        "are sorted"))
end

# The index in the integration arrays of each particle of `system`,
# or `nothing` when particles are not sorted.
@inline function integration_index(system, semi)
    return integration_index(system, semi, particle_sorting(semi))
end

@inline integration_index(system, semi, ::Nothing) = nothing

@inline function integration_index(system, semi, sorting::ParticleSorting)
    return sorting.original_index[system_indices(system, semi)]
end

# The index in the integration arrays of the particle `particle` of a system.
@propagate_inbounds integration_particle(::Nothing, particle) = particle
@propagate_inbounds integration_particle(index, particle) = index[particle]

# Arrays of `system` that are not stored in the integration arrays and have to be reordered
# when the particles are sorted. The last dimension of these arrays must be the particle.
# Arrays that are recomputed in every update after sorting don't need to be included.
#
# To support particle sorting a system must define a method of this function.
# Defining it is a promise that the system works with sorted particles, not just a list
# of arrays. In particular, the system must not rely on the initial particle order,
# for example through stored particle indices (these have to be updated
# in `update_particle_indices!`) or through particles at fixed positions in its arrays.
# Systems without arrays to be reordered return an empty tuple.
function particle_arrays_to_sort(system)
    throw(ArgumentError("sorting particles with a neighborhood search that sorts " *
                        "points is not (yet) supported for systems of type " *
                        "$(nameof(typeof(system)))"))
end

# Update arrays of particle indices of `system` (like the list of moving particles
# of a `PrescribedMotion`) when the particles are sorted.
# `new_index[i]` is the new index of the particle with the previous index `i`.
# `compute_new_index!()` computes `new_index` and returns it. It is only called when needed.
update_particle_indices!(system, compute_new_index!, semi) = system

# Return all arrays in the `NamedTuple` `cache` whose last dimension has length `n`.
function per_particle_arrays(cache, n)
    return Tuple(x for x in values(cache) if x isa AbstractArray && size(x, ndims(x)) == n)
end

# Particles are only sorted with the `CombinedNHSHandler` when its neighborhood search
# sorts points.
uses_particle_sorting(handler::AbstractNHSHandler) = false

function uses_particle_sorting(handler::CombinedNHSHandler)
    return PointNeighbors.sorts_points(first(handler.neighborhood_search.searches))
end

# Adaptor for `Adapt.adapt` that copies all arrays.
struct CopyArrays end

Adapt.adapt_storage(::CopyArrays, x::AbstractArray) = copy(x)

# Create the `ParticleSorting` for `semi` and sort the particles for the first time.
# Return `nothing` when the neighborhood search does not sort points.
# This has to be called after the neighborhood searches have been initialized.
function create_particle_sorting(semi, v_ode, u_ode)
    (; systems) = semi

    uses_particle_sorting(semi.neighborhood_search_handler) || return nothing

    original_index = map(system -> allocate_index(semi, nparticles(system)), systems)
    sorted_index = map(system -> allocate_index(semi, nparticles(system)), systems)
    index_buffer = map(system -> allocate_index(semi, nparticles(system)), systems)
    foreach_system(semi) do system
        # TODO support TLSPH with clamped particles.
        if !(n_integrated_particles(system) in (0, nparticles(system)))
            throw(ArgumentError("sorting particles is only supported when either all or " *
                                "none of the particles of a system are integrated"))
        end
    end
    arrays_to_sort = map(system -> unique_arrays(particle_arrays_to_sort(system)), systems)
    array_buffers = map(arrays -> map(similar, arrays), arrays_to_sort)

    sorting = ParticleSorting(original_index, sorted_index, index_buffer,
                              arrays_to_sort,
                              array_buffers,
                              SortedODEArray(similar(v_ode)),
                              SortedODEArray(similar(u_ode)),
                              SortedODEArray(similar(v_ode)),
                              SortedODEArray(similar(u_ode)))
    semi_new = @set semi.particle_sorting = sorting

    # Sort all systems once with their cell lists, which have been initialized
    # with the unsorted particles. Systems whose cell lists are never updated are never
    # sorted again. There are no sorted copies of the integration arrays to be reordered yet.
    foreach_system(semi_new) do system
        sort_particles_by_nhs!(system, sorting, nothing, nothing, semi_new)
    end

    return sorting
end

function allocate_index(semi, n)
    index = allocate(semi.parallelization_backend, Int, n)
    index .= 1:n

    return index
end

# Remove arrays that appear multiple times, so that they are only reordered once.
function unique_arrays(arrays)
    return Tuple(array
                 for (i, array) in enumerate(arrays)
                 if !any(other -> other === array, arrays[1:(i - 1)]))
end

# Call `f(system, arrays, buffers)` for each system, where `arrays` are the arrays
# to be reordered when the particles are sorted and `buffers` the corresponding buffers.
# When particles are not sorted, `arrays` and `buffers` are `nothing`.
@inline function foreach_system_sorting_arrays(f, semi)
    foreach_system_sorting_arrays(f, semi, particle_sorting(semi))
end

@inline function foreach_system_sorting_arrays(f, semi, ::Nothing)
    foreach_system(system -> f(system, nothing, nothing), semi)
end

@inline function foreach_system_sorting_arrays(f, semi, sorting::ParticleSorting)
    foreach_noalloc_zip(semi.systems,
                        zip_tuples(sorting.arrays_to_sort,
                                   sorting.array_buffers)) do (system, (arrays, buffers))
        f(system, arrays, buffers)
    end
end

@inline zip_tuples(a, b) = map(tuple, a, b)

# `semi.particle_sorting` is `nothing` -> particles are not sorted.
@inline sort_particles_by_nhs!(system, ::Nothing, v_ode, u_ode, semi) = system

# Reorder all arrays of `system` and the sorted copies `v_ode_sorted` and `u_ode_sorted`
# of the integration arrays with the permutation of its cell list, which has been
# updated with the coordinates in the previous system order.
function sort_particles_by_nhs!(system, sorting::ParticleSorting,
                                v_ode_sorted, u_ode_sorted, semi)
    system_index = system_indices(system, semi)
    arrays = sorting.arrays_to_sort[system_index]
    buffers = sorting.array_buffers[system_index]

    @trixi_timeit timer() "sort particles" begin
        sort_particles_by_nhs_inner!(system, sorting, arrays, buffers,
                                     v_ode_sorted, u_ode_sorted, system_index, semi)
    end
end

function sort_particles_by_nhs_inner!(system, sorting, arrays, buffers,
                                      v_ode_sorted, u_ode_sorted, system_index, semi)
    nhs = PointNeighbors.grid_nhs_pair(semi.neighborhood_search_handler.neighborhood_search,
                                       system_index, system_index)

    # `old_index[i]` is the index in the previous system order of the now `i`-th particle.
    old_index = PointNeighbors.point_permutation(nhs)

    reorder_sorted_ode_arrays!(v_ode_sorted, u_ode_sorted, system, old_index, semi)

    original_index = sorting.original_index[system_index]
    sorted_index = sorting.sorted_index[system_index]
    index_buffer = sorting.index_buffer[system_index]

    # Update `original_index`.
    copy_parallel!(index_buffer, original_index, semi)
    @threaded semi for i in eachindex(original_index)
        @inbounds original_index[i] = index_buffer[old_index[i]]
    end

    # Update `sorted_index`.
    @threaded semi for i in eachindex(original_index)
        @inbounds sorted_index[original_index[i]] = i
    end

    # Reorder all system arrays (fields of the system and caches).
    foreach_noalloc_zip(arrays, buffers) do (array, array_buffer)
        reorder_particle_array!(array, old_index, array_buffer, semi)
    end

    # Reorder the system's `SystemBuffer` if used.
    reorder_buffer!(buffer(system), old_index, index_buffer, semi)

    # Update particle index arrays of the system (like the list of moving particles
    # of a `PrescribedMotion`) with the inverse permutation.
    update_particle_indices!(system,
                             () -> invert_permutation!(index_buffer, old_index, semi),
                             semi)

    return system
end

# No integration arrays to reorder (this is the case for the initial sorting).
reorder_sorted_ode_arrays!(::Nothing, ::Nothing, system, old_index, semi) = nothing, nothing

function reorder_sorted_ode_arrays!(v_ode_sorted, u_ode_sorted, system, old_index, semi)
    # Systems without integrated particles have no data in the integration arrays.
    n_integrated_particles(system) == 0 && return nothing

    sorting = particle_sorting(semi)

    # `dv_ode_sorted` is only used in `kick!` after the particles have been sorted
    # and can therefore be used as a buffer here.
    reorder_particle_array!(wrap_v(v_ode_sorted, system, semi), old_index,
                            wrap_v(sorting.dv_ode_sorted, system, semi), semi)
    reorder_particle_array!(wrap_u(u_ode_sorted, system, semi), old_index,
                            wrap_u(sorting.u_ode_buffer, system, semi), semi)

    return v_ode_sorted, u_ode_sorted
end

# `new_index[old_index[i]] = i` for all `i`.
function invert_permutation!(new_index, old_index, semi)
    @threaded semi for i in eachindex(old_index)
        @inbounds new_index[old_index[i]] = i
    end

    return new_index
end

# `array[.., i] = array[.., old_index[i]]` for all `i`.
function reorder_particle_array!(array, old_index, buffer, semi)
    copy_parallel!(buffer, array, semi)

    array_ = reshape(array, :, size(array, ndims(array)))
    buffer_ = reshape(buffer, :, size(buffer, ndims(buffer)))
    @threaded semi for i in axes(array_, 2)
        for j in axes(array_, 1)
            @inbounds array_[j, i] = buffer_[j, old_index[i]]
        end
    end

    return array
end

reorder_buffer!(::Nothing, old_index, index_buffer, semi) = nothing

function reorder_buffer!(buffer::SystemBuffer, old_index, index_buffer, semi)
    (; active_particle) = buffer

    # Use the index buffer as a temporary buffer.
    index_buffer .= active_particle
    @threaded semi for i in eachindex(active_particle)
        @inbounds active_particle[i] = index_buffer[old_index[i]]
    end

    # The inactive particles are sorted to the end by the neighborhood search,
    # so we have to do the same here for the active particle array.
    update_system_buffer!(buffer)
end

# Return the integration arrays in system order.
# When particles are sorted, copy `v_ode` and `u_ode` to the sorted buffers in system order
# and return these. Otherwise, return the original arrays.
# Note that the sorted buffers are not copies. They are overwritten by the next call
# of this function, for example in the next `kick!`.
# See the comments at the top of this file.
function sorted_ode_arrays(v_ode, u_ode, semi)
    return sorted_ode_arrays(v_ode, u_ode, semi, particle_sorting(semi))
end

sorted_ode_arrays(v_ode, u_ode, semi, ::Nothing) = (v_ode, u_ode)

function sorted_ode_arrays(v_ode, u_ode, semi, sorting::ParticleSorting)
    (; v_ode_sorted, u_ode_sorted) = sorting

    # This function returns sorted copies of the raw integration arrays.
    # Passing already sorted copies is likely a mistake.
    if v_ode isa SortedODEArray || u_ode isa SortedODEArray
        throw(ArgumentError("expected the integration arrays, but got sorted copies " *
                            "of the integration arrays"))
    end

    @trixi_timeit timer() "copy to sorted arrays" begin
        copy_to_system_order!(v_ode_sorted, v_ode, wrap_v_unchecked, semi)
        copy_to_system_order!(u_ode_sorted, u_ode, wrap_u_unchecked, semi)
    end

    return v_ode_sorted, u_ode_sorted
end

# Return the buffer for `dv_ode` in system order that `kick!` writes to.
@inline sorted_dv_buffer(dv_ode, semi) = sorted_dv_buffer(dv_ode, particle_sorting(semi))
@inline sorted_dv_buffer(dv_ode, ::Nothing) = dv_ode
@inline sorted_dv_buffer(dv_ode, sorting::ParticleSorting) = sorting.dv_ode_sorted

# Return new copies of `dv_ode` and `du_ode` in system order when particles are sorted.
# Otherwise, return the original arrays.
# This is used by callbacks that call `kick!` to compute the RHS.
function sorted_copies(dv_ode, du_ode, semi)
    return sorted_copies(dv_ode, du_ode, semi, particle_sorting(semi))
end

sorted_copies(dv_ode, du_ode, semi, ::Nothing) = (dv_ode, du_ode)

function sorted_copies(dv_ode, du_ode, semi, ::ParticleSorting)
    dv_ode_sorted = SortedODEArray(similar(dv_ode))
    du_ode_sorted = SortedODEArray(similar(du_ode))

    copy_to_system_order!(dv_ode_sorted, dv_ode, wrap_v_unchecked, semi)
    copy_to_system_order!(du_ode_sorted, du_ode, wrap_u_unchecked, semi)

    return dv_ode_sorted, du_ode_sorted
end

# Copy the integration array `array_ode` to `array_ode_sorted` in system order.
# `wrap` is `wrap_v_unchecked` or `wrap_u_unchecked`.
function copy_to_system_order!(array_ode_sorted, array_ode, wrap, semi)
    sorting = particle_sorting(semi)

    foreach_system(semi) do system
        # The destination is empty for systems without integrated particles.
        original_index = sorting.original_index[system_indices(system, semi)]

        # `wrapped_array_sorted[:, i] .= wrapped_array[:, original_index[i]]`
        copy_permuted!(wrap(array_ode_sorted, system, semi),
                       wrap(array_ode, system, semi), original_index, semi)
    end

    return array_ode_sorted
end

# Copy the integration array `array_ode_sorted` in system order back to `array_ode`
# in integration order.
# `wrap` is `wrap_v_unchecked` or `wrap_u_unchecked`.
function copy_to_integration_order!(array_ode, array_ode_sorted, wrap, semi)
    sorting = particle_sorting(semi)

    foreach_system(semi) do system
        # The destination is empty for systems without integrated particles.
        sorted_index = sorting.sorted_index[system_indices(system, semi)]

        # Gathering from the sorted array,
        # `wrapped_array[:, i] .= wrapped_array_sorted[:, sorted_index[i]]`,
        # is faster on GPUs than scattering to the integration array,
        # `wrapped_array[:, original_index[i]] .= wrapped_array_sorted[:, i]`.
        # This is the only reason why the inverse permutation `sorted_index` is stored.
        copy_permuted!(wrap(array_ode, system, semi),
                       wrap(array_ode_sorted, system, semi), sorted_index, semi)
    end

    return array_ode
end

# Copy the sorted copies `v_ode_sorted` and `u_ode_sorted` back to the integration arrays
# `v_ode` and `u_ode`. This is required when the sorted copies have been modified.
function unsort_ode_arrays!(v_ode, u_ode, v_ode_sorted, u_ode_sorted, semi)
    return unsort_ode_arrays!(v_ode, u_ode, v_ode_sorted, u_ode_sorted, semi,
                              particle_sorting(semi))
end

unsort_ode_arrays!(v_ode, u_ode, v_ode_sorted, u_ode_sorted, semi, ::Nothing) = v_ode, u_ode

function unsort_ode_arrays!(v_ode, u_ode, v_ode_sorted, u_ode_sorted, semi,
                            ::ParticleSorting)
    @trixi_timeit timer() "copy from sorted arrays" begin
        copy_to_integration_order!(v_ode, v_ode_sorted, wrap_v_unchecked, semi)
        copy_to_integration_order!(u_ode, u_ode_sorted, wrap_u_unchecked, semi)
    end

    return v_ode, u_ode
end

# Copy `dv_ode_sorted` from system order back to the integration order of `dv_ode`.
function unsort_dv!(dv_ode, dv_ode_sorted, semi)
    unsort_dv!(dv_ode, dv_ode_sorted, semi, particle_sorting(semi))
end

unsort_dv!(dv_ode, dv_ode_sorted, semi, ::Nothing) = dv_ode

function unsort_dv!(dv_ode, dv_ode_sorted, semi, ::ParticleSorting)
    @trixi_timeit timer() "copy from sorted arrays" begin
        copy_to_integration_order!(dv_ode, dv_ode_sorted, wrap_v_unchecked, semi)
    end

    return dv_ode
end

# Reorder the integration arrays in place to the current system order.
# This is used by the `SortingCallback`, which is the only place where the integration
# arrays are reordered.
function apply_system_order!(v_ode, u_ode, semi)
    sorting = particle_sorting(semi)
    (; v_ode_sorted, u_ode_sorted) = sorting

    # Copy to the sorted arrays first and then back to the integration arrays.
    sorted_ode_arrays(v_ode, u_ode, semi, sorting)
    copyto!(v_ode, parent(v_ode_sorted))
    copyto!(u_ode, parent(u_ode_sorted))

    # Now, the integration order is the system order.
    foreach_noalloc_zip(sorting.original_index,
                        sorting.sorted_index) do (original_index, sorted_index)
        original_index .= eachindex(original_index)
        sorted_index .= eachindex(sorted_index)
    end

    return v_ode, u_ode
end

# `dest[:, i] .= src[:, permutation[i]]`, but in parallel.
function copy_permuted!(dest, src, permutation, semi)
    @threaded semi for i in axes(dest, 2)
        for j in axes(dest, 1)
            @inbounds dest[j, i] = src[j, permutation[i]]
        end
    end

    return dest
end

# `dest .= src`, but in parallel.
function copy_parallel!(dest, src, semi)
    @threaded semi for i in eachindex(dest, src)
        @inbounds dest[i] = src[i]
    end

    return dest
end
