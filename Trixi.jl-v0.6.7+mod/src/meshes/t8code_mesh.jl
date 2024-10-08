"""
    T8codeMesh{NDIMS} <: AbstractMesh{NDIMS}

An unstructured curved mesh based on trees that uses the C library 
['t8code'](https://github.com/DLR-AMR/t8code)
to manage trees and mesh refinement.
"""
mutable struct T8codeMesh{NDIMS, RealT <: Real, IsParallel, NDIMSP2, NNODES} <:
               AbstractMesh{NDIMS}
    cmesh       :: Ptr{t8_cmesh} # cpointer to coarse mesh
    scheme      :: Ptr{t8_eclass_scheme} # cpointer to element scheme
    forest      :: Ptr{t8_forest} # cpointer to forest
    is_parallel :: IsParallel

    # This specifies the geometry interpolation for each tree.
    tree_node_coordinates::Array{RealT, NDIMSP2} # [dimension, i, j, k, tree]

    # Stores the quadrature nodes.
    nodes::SVector{NNODES, RealT}

    boundary_names   :: Array{Symbol, 2}      # [face direction, tree]
    current_filename :: String

    ninterfaces :: Int
    nmortars    :: Int
    nboundaries :: Int

    function T8codeMesh{NDIMS}(cmesh, scheme, forest, tree_node_coordinates, nodes,
                               boundary_names,
                               current_filename) where {NDIMS}
        is_parallel = False()

        mesh = new{NDIMS, Float64, typeof(is_parallel), NDIMS + 2, length(nodes)}(cmesh,
                                                                                  scheme,
                                                                                  forest,
                                                                                  is_parallel)

        mesh.nodes = nodes
        mesh.boundary_names = boundary_names
        mesh.current_filename = current_filename
        mesh.tree_node_coordinates = tree_node_coordinates

        finalizer(mesh) do mesh
            # When finalizing `mesh.forest`, `mesh.scheme` and `mesh.cmesh` are
            # also cleaned up from within `t8code`. The cleanup code for
            # `cmesh` does some MPI calls for deallocating shared memory
            # arrays. Due to garbage collection in Julia the order of shutdown
            # is not deterministic. The following code might happen after MPI
            # is already in finalized state.
            # If the environment variable `TRIXI_T8CODE_SC_FINALIZE` is set the
            # `finalize_hook` of the MPI module takes care of the cleanup. See
            # further down. However, this might cause a pile-up of `mesh`
            # objects during long-running sessions.
            if !MPI.Finalized()
                trixi_t8_unref_forest(mesh.forest)
            end
        end

        # This finalizer call is only recommended during development and not for
        # production runs, especially long-running sessions since a reference to
        # the `mesh` object will be kept throughout the lifetime of the session.
        # See comments in `init_t8code()` in file `src/auxiliary/t8code.jl` for
        # more information.
        if haskey(ENV, "TRIXI_T8CODE_SC_FINALIZE")
            MPI.add_finalize_hook!() do
                trixi_t8_unref_forest(mesh.forest)
            end
        end

        return mesh
    end
end

const SerialT8codeMesh{NDIMS} = T8codeMesh{NDIMS, <:Real, <:False}
@inline mpi_parallel(mesh::SerialT8codeMesh) = False()

@inline Base.ndims(::T8codeMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::T8codeMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline ntrees(mesh::T8codeMesh) = Int(t8_forest_get_num_local_trees(mesh.forest))
@inline ncells(mesh::T8codeMesh) = Int(t8_forest_get_local_num_elements(mesh.forest))
@inline ninterfaces(mesh::T8codeMesh) = mesh.ninterfaces
@inline nmortars(mesh::T8codeMesh) = mesh.nmortars
@inline nboundaries(mesh::T8codeMesh) = mesh.nboundaries

function Base.show(io::IO, mesh::T8codeMesh)
    print(io, "T8codeMesh{", ndims(mesh), ", ", real(mesh), "}")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::T8codeMesh)
    if get(io, :compact, false)
        show(io, mesh)
    else
        setup = [
            "#trees" => ntrees(mesh),
            "current #cells" => ncells(mesh),
            "polydeg" => length(mesh.nodes) - 1
        ]
        summary_box(io,
                    "T8codeMesh{" * string(ndims(mesh)) * ", " * string(real(mesh)) * "}",
                    setup)
    end
end

"""
    T8codeMesh(trees_per_dimension; polydeg, mapping=identity,
               RealT=Float64, initial_refinement_level=0, periodicity=true)

Create a structured potentially curved 'T8codeMesh' of the specified size.

Non-periodic boundaries will be called ':x_neg', ':x_pos', ':y_neg', ':y_pos', ':z_neg', ':z_pos'.

# Arguments
- 'trees_per_dimension::NTupleE{NDIMS, Int}': the number of trees in each dimension.
- 'polydeg::Integer': polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
- `mapping`: a function of `NDIMS` variables to describe the mapping that transforms
             the reference mesh (`[-1, 1]^n`) to the physical domain.
             Use only one of `mapping`, `faces` and `coordinates_min`/`coordinates_max`.
- `faces::NTuple{2*NDIMS}`: a tuple of `2 * NDIMS` functions that describe the faces of the domain.
                            Each function must take `NDIMS-1` arguments.
                            `faces[1]` describes the face onto which the face in negative x-direction
                            of the unit hypercube is mapped. The face in positive x-direction of
                            the unit hypercube will be mapped onto the face described by `faces[2]`.
                            `faces[3:4]` describe the faces in positive and negative y-direction respectively
                            (in 2D and 3D).
                            `faces[5:6]` describe the faces in positive and negative z-direction respectively (in 3D).
                            Use only one of `mapping`, `faces` and `coordinates_min`/`coordinates_max`.
- `coordinates_min`: vector or tuple of the coordinates of the corner in the negative direction of each dimension
                     to create a rectangular mesh.
                     Use only one of `mapping`, `faces` and `coordinates_min`/`coordinates_max`.
- `coordinates_max`: vector or tuple of the coordinates of the corner in the positive direction of each dimension
                     to create a rectangular mesh.
                     Use only one of `mapping`, `faces` and `coordinates_min`/`coordinates_max`.
- 'RealT::Type': the type that should be used for coordinates.
- 'initial_refinement_level::Integer': refine the mesh uniformly to this level before the simulation starts.
- 'periodicity': either a 'Bool' deciding if all of the boundaries are periodic or an 'NTuple{NDIMS, Bool}'
                 deciding for each dimension if the boundaries in this dimension are periodic.
"""
function T8codeMesh(trees_per_dimension; polydeg = 1,
                    mapping = nothing, faces = nothing, coordinates_min = nothing,
                    coordinates_max = nothing,
                    RealT = Float64, initial_refinement_level = 0,
                    periodicity = true)
    @assert ((coordinates_min === nothing)===(coordinates_max === nothing)) "Either both or none of coordinates_min and coordinates_max must be specified"

    @assert count(i -> i !== nothing,
                  (mapping, faces, coordinates_min))==1 "Exactly one of mapping, faces and coordinates_min/max must be specified"

    # Extract mapping
    if faces !== nothing
        validate_faces(faces)
        mapping = transfinite_mapping(faces)
    elseif coordinates_min !== nothing
        mapping = coordinates2mapping(coordinates_min, coordinates_max)
    end

    NDIMS = length(trees_per_dimension)
    @assert (NDIMS == 2||NDIMS == 3) "NDIMS should be 2 or 3."

    # Convert periodicity to a Tuple of a Bool for every dimension
    if all(periodicity)
        # Also catches case where periodicity = true
        periodicity = ntuple(_ -> true, NDIMS)
    elseif !any(periodicity)
        # Also catches case where periodicity = false
        periodicity = ntuple(_ -> false, NDIMS)
    else
        # Default case if periodicity is an iterable
        periodicity = Tuple(periodicity)
    end

    do_partition = 0
    if NDIMS == 2
        conn = T8code.Libt8.p4est_connectivity_new_brick(trees_per_dimension...,
                                                         periodicity...)
        cmesh = t8_cmesh_new_from_p4est(conn, mpi_comm(), do_partition)
        T8code.Libt8.p4est_connectivity_destroy(conn)
    elseif NDIMS == 3
        conn = T8code.Libt8.p8est_connectivity_new_brick(trees_per_dimension...,
                                                         periodicity...)
        cmesh = t8_cmesh_new_from_p8est(conn, mpi_comm(), do_partition)
        T8code.Libt8.p8est_connectivity_destroy(conn)
    end

    scheme = t8_scheme_new_default_cxx()
    forest = t8_forest_new_uniform(cmesh, scheme, initial_refinement_level, 0, mpi_comm())

    basis = LobattoLegendreBasis(RealT, polydeg)
    nodes = basis.nodes

    tree_node_coordinates = Array{RealT, NDIMS + 2}(undef, NDIMS,
                                                    ntuple(_ -> length(nodes), NDIMS)...,
                                                    prod(trees_per_dimension))

    # Get cell length in reference mesh: Omega_ref = [-1,1]^NDIMS.
    dx = [2 / n for n in trees_per_dimension]

    num_local_trees = t8_cmesh_get_num_local_trees(cmesh)

    # Non-periodic boundaries.
    boundary_names = fill(Symbol("---"), 2 * NDIMS, prod(trees_per_dimension))

    if mapping === nothing
        mapping_ = coordinates2mapping(ntuple(_ -> -1.0, NDIMS), ntuple(_ -> 1.0, NDIMS))
    else
        mapping_ = mapping
    end

    for itree in 1:num_local_trees
        veptr = t8_cmesh_get_tree_vertices(cmesh, itree - 1)
        verts = unsafe_wrap(Array, veptr, (3, 1 << NDIMS))

        # Calculate node coordinates of reference mesh.
        if NDIMS == 2
            cell_x_offset = (verts[1, 1] - 0.5 * (trees_per_dimension[1] - 1)) * dx[1]
            cell_y_offset = (verts[2, 1] - 0.5 * (trees_per_dimension[2] - 1)) * dx[2]

            for j in eachindex(nodes), i in eachindex(nodes)
                tree_node_coordinates[:, i, j, itree] .= mapping_(cell_x_offset +
                                                                  dx[1] * nodes[i] / 2,
                                                                  cell_y_offset +
                                                                  dx[2] * nodes[j] / 2)
            end
        elseif NDIMS == 3
            cell_x_offset = (verts[1, 1] - 0.5 * (trees_per_dimension[1] - 1)) * dx[1]
            cell_y_offset = (verts[2, 1] - 0.5 * (trees_per_dimension[2] - 1)) * dx[2]
            cell_z_offset = (verts[3, 1] - 0.5 * (trees_per_dimension[3] - 1)) * dx[3]

            for k in eachindex(nodes), j in eachindex(nodes), i in eachindex(nodes)
                tree_node_coordinates[:, i, j, k, itree] .= mapping_(cell_x_offset +
                                                                     dx[1] * nodes[i] / 2,
                                                                     cell_y_offset +
                                                                     dx[2] * nodes[j] / 2,
                                                                     cell_z_offset +
                                                                     dx[3] * nodes[k] / 2)
            end
        end

        if !periodicity[1]
            boundary_names[1, itree] = :x_neg
            boundary_names[2, itree] = :x_pos
        end

        if !periodicity[2]
            boundary_names[3, itree] = :y_neg
            boundary_names[4, itree] = :y_pos
        end

        if NDIMS > 2
            if !periodicity[3]
                boundary_names[5, itree] = :z_neg
                boundary_names[6, itree] = :z_pos
            end
        end
    end

    return T8codeMesh{NDIMS}(cmesh, scheme, forest, tree_node_coordinates, nodes,
                             boundary_names, "")
end

"""
    T8codeMesh(cmesh::Ptr{t8_cmesh},
               mapping=nothing, polydeg=1, RealT=Float64,
               initial_refinement_level=0)

Main mesh constructor for the `T8codeMesh` that imports an unstructured,
conforming mesh from a `t8_cmesh` data structure.

# Arguments
- `cmesh::Ptr{t8_cmesh}`: Pointer to a cmesh object.
- `mapping`: a function of `NDIMS` variables to describe the mapping that transforms
             the imported mesh to the physical domain. Use `nothing` for the identity map.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
                      The default of `1` creates an uncurved geometry. Use a higher value if the mapping
                      will curve the imported uncurved mesh.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
"""
function T8codeMesh(cmesh::Ptr{t8_cmesh};
                    mapping = nothing, polydeg = 1, RealT = Float64,
                    initial_refinement_level = 0)
    @assert (t8_cmesh_get_num_trees(cmesh)>0) "Given `cmesh` does not contain any trees."

    # Infer NDIMS from the geometry of the first tree.
    NDIMS = Int(t8_geom_get_dimension(t8_cmesh_get_tree_geometry(cmesh, 0)))

    @assert (NDIMS == 2||NDIMS == 3) "NDIMS should be 2 or 3."

    scheme = t8_scheme_new_default_cxx()
    forest = t8_forest_new_uniform(cmesh, scheme, initial_refinement_level, 0, mpi_comm())

    basis = LobattoLegendreBasis(RealT, polydeg)
    nodes = basis.nodes

    num_local_trees = t8_cmesh_get_num_local_trees(cmesh)

    tree_node_coordinates = Array{RealT, NDIMS + 2}(undef, NDIMS,
                                                    ntuple(_ -> length(nodes), NDIMS)...,
                                                    num_local_trees)

    nodes_in = [-1.0, 1.0]
    matrix = polynomial_interpolation_matrix(nodes_in, nodes)

    if NDIMS == 2
        data_in = Array{RealT, 3}(undef, 2, 2, 2)
        tmp1 = zeros(RealT, 2, length(nodes), length(nodes_in))
        verts = zeros(3, 4)

        for itree in 0:(num_local_trees - 1)
            veptr = t8_cmesh_get_tree_vertices(cmesh, itree)

            # Note, `verts = unsafe_wrap(Array, veptr, (3, 1 << NDIMS))`
            # sometimes does not work since `veptr` is not necessarily properly
            # aligned to 8 bytes.
            for icorner in 1:4
                verts[1, icorner] = unsafe_load(veptr, (icorner - 1) * 3 + 1)
                verts[2, icorner] = unsafe_load(veptr, (icorner - 1) * 3 + 2)
            end

            # Check if tree's node ordering is right-handed or print a warning.
            let z = zero(eltype(verts)), o = one(eltype(verts))
                u = verts[:, 2] - verts[:, 1]
                v = verts[:, 3] - verts[:, 1]
                w = [z, z, o]

                # Triple product gives signed volume of spanned parallelepiped.
                vol = dot(cross(u, v), w)

                if vol < z
                    @warn "Discovered negative volumes in `cmesh`: vol = $vol"
                end
            end

            # Tree vertices are stored in z-order.
            @views data_in[:, 1, 1] .= verts[1:2, 1]
            @views data_in[:, 2, 1] .= verts[1:2, 2]
            @views data_in[:, 1, 2] .= verts[1:2, 3]
            @views data_in[:, 2, 2] .= verts[1:2, 4]

            # Interpolate corner coordinates to specified nodes.
            multiply_dimensionwise!(view(tree_node_coordinates, :, :, :, itree + 1),
                                    matrix, matrix,
                                    data_in,
                                    tmp1)
        end

    elseif NDIMS == 3
        data_in = Array{RealT, 4}(undef, 3, 2, 2, 2)
        tmp1 = zeros(RealT, 3, length(nodes), length(nodes_in), length(nodes_in))
        verts = zeros(3, 8)

        for itree in 0:(num_local_trees - 1)
            veptr = t8_cmesh_get_tree_vertices(cmesh, itree)

            # Note, `verts = unsafe_wrap(Array, veptr, (3, 1 << NDIMS))`
            # sometimes does not work since `veptr` is not necessarily properly
            # aligned to 8 bytes.
            for icorner in 1:8
                verts[1, icorner] = unsafe_load(veptr, (icorner - 1) * 3 + 1)
                verts[2, icorner] = unsafe_load(veptr, (icorner - 1) * 3 + 2)
                verts[3, icorner] = unsafe_load(veptr, (icorner - 1) * 3 + 3)
            end

            # Tree vertices are stored in z-order.
            @views data_in[:, 1, 1, 1] .= verts[1:3, 1]
            @views data_in[:, 2, 1, 1] .= verts[1:3, 2]
            @views data_in[:, 1, 2, 1] .= verts[1:3, 3]
            @views data_in[:, 2, 2, 1] .= verts[1:3, 4]

            @views data_in[:, 1, 1, 2] .= verts[1:3, 5]
            @views data_in[:, 2, 1, 2] .= verts[1:3, 6]
            @views data_in[:, 1, 2, 2] .= verts[1:3, 7]
            @views data_in[:, 2, 2, 2] .= verts[1:3, 8]

            # Interpolate corner coordinates to specified nodes.
            multiply_dimensionwise!(view(tree_node_coordinates, :, :, :, :, itree + 1),
                                    matrix, matrix, matrix,
                                    data_in,
                                    tmp1)
        end
    end

    map_node_coordinates!(tree_node_coordinates, mapping)

    # There's no simple and generic way to distinguish boundaries. Name all of them :all.
    boundary_names = fill(:all, 2 * NDIMS, num_local_trees)

    return T8codeMesh{NDIMS}(cmesh, scheme, forest, tree_node_coordinates, nodes,
                             boundary_names, "")
end

"""
    T8codeMesh(conn::Ptr{p4est_connectivity}; kwargs...)

Main mesh constructor for the `T8codeMesh` that imports an unstructured,
conforming mesh from a `p4est_connectivity` data structure.

# Arguments
- `conn::Ptr{p4est_connectivity}`: Pointer to a P4est connectivity object.
- `mapping`: a function of `NDIMS` variables to describe the mapping that transforms
             the imported mesh to the physical domain. Use `nothing` for the identity map.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
                      The default of `1` creates an uncurved geometry. Use a higher value if the mapping
                      will curve the imported uncurved mesh.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
"""
function T8codeMesh(conn::Ptr{p4est_connectivity}; kwargs...)
    cmesh = t8_cmesh_new_from_p4est(conn, mpi_comm(), 0)

    return T8codeMesh(cmesh; kwargs...)
end

"""
    T8codeMesh(conn::Ptr{p8est_connectivity}; kwargs...)

Main mesh constructor for the `T8codeMesh` that imports an unstructured,
conforming mesh from a `p4est_connectivity` data structure.

# Arguments
- `conn::Ptr{p4est_connectivity}`: Pointer to a P4est connectivity object.
- `mapping`: a function of `NDIMS` variables to describe the mapping that transforms
             the imported mesh to the physical domain. Use `nothing` for the identity map.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
                      The default of `1` creates an uncurved geometry. Use a higher value if the mapping
                      will curve the imported uncurved mesh.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
"""
function T8codeMesh(conn::Ptr{p8est_connectivity}; kwargs...)
    cmesh = t8_cmesh_new_from_p8est(conn, mpi_comm(), 0)

    return T8codeMesh(cmesh; kwargs...)
end

"""
    T8codeMesh{NDIMS}(meshfile::String; kwargs...)

Main mesh constructor for the `T8codeMesh` that imports an unstructured, conforming
mesh from a Gmsh mesh file (`.msh`).

# Arguments
- `meshfile::String`: path to a Gmsh mesh file.
- `ndims`: Mesh file dimension: `2` or `3`.
- `mapping`: a function of `NDIMS` variables to describe the mapping that transforms
             the imported mesh to the physical domain. Use `nothing` for the identity map.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
                      The default of `1` creates an uncurved geometry. Use a higher value if the mapping
                      will curve the imported uncurved mesh.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
"""
function T8codeMesh(meshfile::String, ndims; kwargs...)

    # Prevent `t8code` from crashing Julia if the file doesn't exist.
    @assert isfile(meshfile)

    meshfile_prefix, meshfile_suffix = splitext(meshfile)

    cmesh = t8_cmesh_from_msh_file(meshfile_prefix, 0, mpi_comm(), ndims, 0, 0)

    return T8codeMesh(cmesh; kwargs...)
end

struct adapt_callback_passthrough
    adapt_callback::Function
    user_data::Any
end

# Callback function prototype to decide for refining and coarsening.
# If `is_family` equals 1, the first `num_elements` in elements
# form a family and we decide whether this family should be coarsened
# or only the first element should be refined.
# Otherwise `is_family` must equal zero and we consider the first entry
# of the element array for refinement. 
# Entries of the element array beyond the first `num_elements` are undefined.
# \param [in] forest       the forest to which the new elements belong
# \param [in] forest_from  the forest that is adapted.
# \param [in] which_tree   the local tree containing `elements`
# \param [in] lelement_id  the local element id in `forest_old` in the tree of the current element
# \param [in] ts           the eclass scheme of the tree
# \param [in] is_family    if 1, the first `num_elements` entries in `elements` form a family. If 0, they do not.
# \param [in] num_elements the number of entries in `elements` that are defined
# \param [in] elements     Pointers to a family or, if `is_family` is zero,
#                          pointer to one element.
# \return greater zero if the first entry in `elements` should be refined,
#         smaller zero if the family `elements` shall be coarsened,
#         zero else.
function adapt_callback_wrapper(forest,
                                forest_from,
                                which_tree,
                                lelement_id,
                                ts,
                                is_family,
                                num_elements,
                                elements_ptr)::Cint
    passthrough = unsafe_pointer_to_objref(t8_forest_get_user_data(forest))[]

    elements = unsafe_wrap(Array, elements_ptr, num_elements)

    return passthrough.adapt_callback(forest_from, which_tree, ts, lelement_id, elements,
                                      Bool(is_family), passthrough.user_data)
end

"""
    Trixi.adapt!(mesh::T8codeMesh, adapt_callback; kwargs...)

Adapt a `T8codeMesh` according to a user-defined `adapt_callback`.

# Arguments
- `mesh::T8codeMesh`: Initialized mesh object.
- `adapt_callback`: A user-defined callback which tells the adaption routines
                    if an element should be refined, coarsened or stay unchanged.

    The expected callback signature is as follows:

      `adapt_callback(forest, ltreeid, eclass_scheme, lelemntid, elements, is_family, user_data)`
        # Arguments
        - `forest`: Pointer to the analyzed forest.
        - `ltreeid`: Local index of the current tree where the analyzed elements are part of.
        - `eclass_scheme`: Element class of `elements`.
        - `lelemntid`: Local index of the first element in `elements`.
        - `elements`: Array of elements. If consecutive elements form a family
                      they are passed together, otherwise `elements` consists of just one element.
        - `is_family`: Boolean signifying if `elements` represents a family or not.
        - `user_data`: Void pointer to some arbitrary user data. Default value is `C_NULL`.
        # Returns
          -1 : Coarsen family of elements.
           0 : Stay unchanged.
           1 : Refine element.

- `kwargs`: 
    - `recursive = true`: Adapt the forest recursively. If true the caller must ensure that the callback 
                          returns 0 for every analyzed element at some point to stop the recursion.
    - `balance = true`: Make sure the adapted forest is 2^(NDIMS-1):1 balanced.
    - `partition = true`: Partition the forest to redistribute elements evenly among MPI ranks.
    - `ghost = true`: Create a ghost layer for MPI data exchange.
    - `user_data = C_NULL`: Pointer to some arbitrary user-defined data.
"""
function adapt!(mesh::T8codeMesh, adapt_callback; recursive = true, balance = true,
                partition = true, ghost = true, user_data = C_NULL)
    # Check that forest is a committed, that is valid and usable, forest.
    @assert t8_forest_is_committed(mesh.forest) != 0

    # Init new forest.
    new_forest_ref = Ref{t8_forest_t}()
    t8_forest_init(new_forest_ref)
    new_forest = new_forest_ref[]

    # Check out `examples/t8_step4_partition_balance_ghost.jl` in
    # https://github.com/DLR-AMR/T8code.jl for detailed explanations.
    let set_from = C_NULL, set_for_coarsening = 0, no_repartition = !partition
        t8_forest_set_user_data(new_forest,
                                pointer_from_objref(Ref(adapt_callback_passthrough(adapt_callback,
                                                                                   user_data))))
        t8_forest_set_adapt(new_forest, mesh.forest,
                            @t8_adapt_callback(adapt_callback_wrapper),
                            recursive)
        if balance
            t8_forest_set_balance(new_forest, set_from, no_repartition)
        end

        if partition
            t8_forest_set_partition(new_forest, set_from, set_for_coarsening)
        end

        t8_forest_set_ghost(new_forest, ghost, T8_GHOST_FACES) # Note: MPI support not available yet so it is a dummy call.

        # The old forest is destroyed here.
        # Call `t8_forest_ref(Ref(mesh.forest))` to keep it.
        t8_forest_commit(new_forest)
    end

    mesh.forest = new_forest

    return nothing
end

# TODO: Just a placeholder. Will be implemented later when MPI is supported.
function balance!(mesh::T8codeMesh, init_fn = C_NULL)
    return nothing
end

# TODO: Just a placeholder. Will be implemented later when MPI is supported.
function partition!(mesh::T8codeMesh; allow_coarsening = true, weight_fn = C_NULL)
    return nothing
end

#! format: off
@deprecate T8codeMesh{2}(conn::Ptr{p4est_connectivity}; kwargs...) T8codeMesh(conn::Ptr{p4est_connectivity}; kwargs...)
@deprecate T8codeMesh{3}(conn::Ptr{p8est_connectivity}; kwargs...) T8codeMesh(conn::Ptr{p8est_connectivity}; kwargs...)
@deprecate T8codeMesh{2}(meshfile::String; kwargs...) T8codeMesh(meshfile::String, 2; kwargs...)
@deprecate T8codeMesh{3}(meshfile::String; kwargs...) T8codeMesh(meshfile::String, 3; kwargs...)
#! format: on
