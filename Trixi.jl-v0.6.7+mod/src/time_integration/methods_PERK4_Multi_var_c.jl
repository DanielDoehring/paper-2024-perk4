# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct PERK4_Multi_var_c{StageCallbacks}
    const NumStageEvalsMin::Int64
    const NumMethods::Int64
    const NumStages::Int64
    const dtRatios::Vector{Float64}
    stage_callbacks::StageCallbacks

    AMatrices::Array{Float64, 3}
    AMatrix::Matrix{Float64}
    c::Vector{Float64}
    ActiveLevels::Vector{Vector{Int64}}
    HighestActiveLevels::Vector{Int64}
    HighestEvalLevels::Vector{Int64}

    function PERK4_Multi_var_c(Stages_::Vector{Int64},
                         BasePathMonCoeffs_::AbstractString,
                         dtRatios_,
                         stage_callbacks = ())
        newPERK4_Multi = new{typeof(stage_callbacks)}(minimum(Stages_),
                                                      length(Stages_),
                                                      maximum(Stages_),
                                                      dtRatios_,
                                                      stage_callbacks)

        newPERK4_Multi.AMatrices, newPERK4_Multi.AMatrix, newPERK4_Multi.c,
        newPERK4_Multi.ActiveLevels, newPERK4_Multi.HighestActiveLevels, newPERK4_Multi.HighestEvalLevels = ComputePERK4_Multi_ButcherTableau(Stages_,
                                                                                                                                              newPERK4_Multi.NumStages,
                                                                                                                                              BasePathMonCoeffs_)

        return newPERK4_Multi
    end
end # struct PERK4_Multi

function init(ode::ODEProblem, alg::PERK4_Multi_var_c;
              dt, callback = nothing, kwargs...)

    u0 = copy(ode.u0)
    du = zero(u0) # previously: similar(u0)
    u_tmp = zero(u0)

    # PERK4_Multi stages
    k1 = zero(u0)
    k_higher = zero(u0)

    t0 = first(ode.tspan)
    iter = 0

    ### Set datastructures for handling of level-dependent integration ###
    mesh, equations, dg, cache = mesh_equations_solver_cache(ode.p)

    n_levels = get_n_levels(mesh, alg)
    n_dims = ndims(mesh) # Spatial dimension

    level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_boundaries_orientation_acc = [[Vector{Int64}()
                                              for _ in 1:(2 * n_dims)]
                                             for _ in 1:n_levels]

    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    
    partitioning_variables!(level_info_elements, 
                            level_info_elements_acc, 
                            level_info_interfaces_acc, 
                            level_info_boundaries_acc, 
                            level_info_boundaries_orientation_acc,
                            level_info_mortars_acc,
                            n_levels, n_dims, mesh, dg, cache, alg,
                            # NOTE: For variable c case:
                            Trixi.wrap_array(u0, ode.p), equations)

    for i in 1:n_levels
        println("#Number Elements integrated with level $i: ", length(level_info_elements[i]))
    end

    # Set initial distribution of DG Base function coefficients
    level_u_indices_elements = [Vector{Int64}() for _ in 1:n_levels]
    partitioning_u!(level_u_indices_elements, n_levels, n_dims, level_info_elements, u0, mesh, equations, dg, cache)

    ### Done with setting up for handling of level-dependent integration ###

    integrator = PERK4_Multi_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                                        (prob = ode,), ode.f, alg,
                                        PERK_IntegratorOptions(callback, ode.tspan;
                                                               kwargs...), false,
                                        k1, k_higher,
                                        
                                        level_info_elements, level_info_elements_acc,

                                        level_info_interfaces_acc,

                                        level_info_boundaries_acc,
                                        level_info_boundaries_orientation_acc,

                                        level_info_mortars_acc,

                                        level_u_indices_elements,
                                        
                                        t0, -1, n_levels,
                                        0.0)

    # initialize callbacks
    if callback isa CallbackSet
        for cb in callback.continuous_callbacks
            error("unsupported")
        end
        for cb in callback.discrete_callbacks
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    elseif !isnothing(callback)
        error("unsupported")
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PERK4_Multi_var_c;
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve_steps!(integrator)
end
end # @muladd
