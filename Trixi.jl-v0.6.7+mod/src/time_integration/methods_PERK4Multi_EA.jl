# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct PERK4_Multi_EA{StageCallbacks}
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

    function PERK4_Multi_EA(Stages_::Vector{Int64},
                         BasePathMonCoeffs_::AbstractString,
                         dtRatios_,
                         stage_callbacks = ())
        newPERK4_Multi_EA = new{typeof(stage_callbacks)}(minimum(Stages_),
                                                      length(Stages_),
                                                      maximum(Stages_),
                                                      dtRatios_,
                                                      stage_callbacks)

        newPERK4_Multi_EA.AMatrices, newPERK4_Multi_EA.AMatrix, newPERK4_Multi_EA.c,
        newPERK4_Multi_EA.ActiveLevels, newPERK4_Multi_EA.HighestActiveLevels, newPERK4_Multi_EA.HighestEvalLevels = ComputePERK4_Multi_ButcherTableau(Stages_,
                                                                                                                                              newPERK4_Multi_EA.NumStages,
                                                                                                                                              BasePathMonCoeffs_)

        return newPERK4_Multi_EA
    end
end # struct PERK4_Multi_EA

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK4_Multi_EA_Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                      PERK_IntegratorOptions}
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    dt::RealT # current time step
    dtcache::RealT # Used for euler-acoustic coupling
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::Alg # This is our own class written above; Abbreviation for ALGorithm
    opts::PERK_IntegratorOptions
    finalstep::Bool # added for convenience
    # PERK4_Multi_EA stages:
    k1::uType
    k_higher::uType
    
    # Variables managing level-depending integration
    level_info_elements::Vector{Vector{Int64}}
    level_info_elements_acc::Vector{Vector{Int64}}

    level_info_interfaces_acc::Vector{Vector{Int64}}
    level_info_mpi_interfaces_acc::Vector{Vector{Int64}}

    level_info_boundaries_acc::Vector{Vector{Int64}}
    level_info_boundaries_orientation_acc::Vector{Vector{Vector{Int64}}}

    level_info_mortars_acc::Vector{Vector{Int64}}
    level_info_mpi_mortars_acc::Vector{Vector{Int64}}

    level_u_indices_elements::Vector{Vector{Int64}}
    
    t_stage::RealT
    coarsest_lvl::Int64
    n_levels::Int64

    # TODO uprev, tprev for averaging callback (required for coupled Euler-acoustic simulations)
    uprev::uType
    tprev::RealT
    
    AddRHSCalls::Float64
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PERK4_Multi_EA_Integrator, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

function init(ode::ODEProblem, alg::PERK4_Multi_EA;
              dt, callback = nothing, kwargs...)

    u0 = copy(ode.u0)
    du = zero(u0) # previously: similar(u0)
    u_tmp = zero(u0)

    # PERK4_Multi_EA stages
    k1 = zero(u0)
    k_higher = zero(u0)

    # TODO: Only for averaging callback (required for coupled Euler-acoustic simulations)
    uprev = zero(u0)
    tprev = zero(ode.tspan[1])

    t0 = first(ode.tspan)
    iter = 0

    ### Set datastructures for handling of level-dependent integration ###
    mesh, equations, dg, cache = mesh_equations_solver_cache(ode.p)

    n_levels = get_n_levels(mesh, alg)
    n_dims = ndims(mesh) # Spatial dimension

    level_info_elements = [Vector{Int64}() for _ in 1:n_levels]
    level_info_elements_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_mpi_interfaces_acc = [Vector{Int64}() for _ in 1:n_levels]

    level_info_boundaries_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_boundaries_orientation_acc = [[Vector{Int64}()
                                              for _ in 1:(2 * n_dims)]
                                             for _ in 1:n_levels]

    level_info_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]
    level_info_mpi_mortars_acc = [Vector{Int64}() for _ in 1:n_levels]

    
    partitioning_variables!(level_info_elements, 
                            level_info_elements_acc, 
                            level_info_interfaces_acc, 
                            level_info_boundaries_acc, 
                            level_info_boundaries_orientation_acc,
                            level_info_mortars_acc,
                            n_levels, n_dims, mesh, dg, cache, alg)


    for i in 1:n_levels
        println("#Number Elements integrated with level $i: ", length(level_info_elements[i]))
    end

    # Set initial distribution of DG Base function coefficients
    level_u_indices_elements = [Vector{Int64}() for _ in 1:n_levels]
    partitioning_u!(level_u_indices_elements, n_levels, n_dims, level_info_elements, u0, mesh, equations, dg, cache)

    ### Done with setting up for handling of level-dependent integration ###

    integrator = PERK4_Multi_EA_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                                        (prob = ode,), ode.f, alg,
                                        PERK_IntegratorOptions(callback, ode.tspan;
                                                               kwargs...), false,
                                        k1, k_higher,
                                        
                                        level_info_elements, level_info_elements_acc,

                                        level_info_interfaces_acc,
                                        level_info_mpi_interfaces_acc,

                                        level_info_boundaries_acc,
                                        level_info_boundaries_orientation_acc,

                                        level_info_mortars_acc,
                                        level_info_mpi_mortars_acc,

                                        level_u_indices_elements,
                                        
                                        t0, -1, n_levels,
                                        uprev, tprev,
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
function solve(ode::ODEProblem, alg::PERK4_Multi_EA;
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve_steps!(integrator)
end

function solve_steps!(integrator::PERK4_Multi_EA_Integrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        # NOTE: `prev` For EulerAcoustics only
        @threaded for u_ind in eachindex(integrator.u)
            integrator.uprev[u_ind] = integrator.u[u_ind]
        end
        integrator.tprev = integrator.t

        step!(integrator)
    end # "main loop" timer

    println("Additional RHS Calls: ", integrator.AddRHSCalls)

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

function k1!(integrator::PERK4_Multi_EA_Integrator, p, c)
    integrator.f(integrator.du, integrator.u, p, integrator.t)

    @threaded for i in eachindex(integrator.du)
        integrator.k1[i] = integrator.du[i] * integrator.dt
    end

    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] + c[2] * integrator.k1[i]
    end

    # TODO: Move away from here, not really belonging to stage 1!
    integrator.t_stage = integrator.t + c[2] * integrator.dt
end

function last_three_stages!(integrator::PERK4_Multi_EA_Integrator, alg, p)
    for stage in 1:2
        @threaded for u_ind in eachindex(integrator.u)
            integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                      alg.AMatrix[stage, 1] *
                                      integrator.k1[u_ind] +
                                      alg.AMatrix[stage, 2] *
                                      integrator.k_higher[u_ind]
        end
        integrator.t_stage = integrator.t +
                             alg.c[alg.NumStages - 3 + stage] * integrator.dt

        integrator.f(integrator.du, integrator.u_tmp, p, integrator.t_stage)

        @threaded for u_ind in eachindex(integrator.du)
            integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
        end
    end

    # Last stage
    @threaded for i in eachindex(integrator.du)
        integrator.u_tmp[i] = integrator.u[i] +
                              alg.AMatrix[3, 1] *
                              integrator.k1[i] +
                              alg.AMatrix[3, 2] *
                              integrator.k_higher[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, p, integrator.t + alg.c[alg.NumStages] * integrator.dt)
    
    @threaded for u_ind in eachindex(integrator.u)
        # "Own" PairedExplicitRK based on SSPRK33.
          # Note that 'k_higher' carries the values of K_{S-1}
          # and that we construct 'K_S' "in-place" from 'integrator.du'
        integrator.u[u_ind] += 0.5 * (integrator.k_higher[u_ind] + integrator.du[u_ind] * integrator.dt)
    end
end

function step!(integrator::PERK4_Multi_EA_Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        k1!(integrator, prob.p, alg.c)
        
        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, 
                        integrator.level_info_elements_acc[1],
                        integrator.level_info_interfaces_acc[1],
                        integrator.level_info_boundaries_acc[1],
                        integrator.level_info_boundaries_orientation_acc[1],
                        integrator.level_info_mortars_acc[1],
                        1)

        # Update finest level only
        @threaded for u_ind in integrator.level_u_indices_elements[1]
            integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
        end

        for stage in 3:(alg.NumStages - 3) 
            ### General implementation: Not own method for each grid level ###
            # Loop over different methods with own associated level
            for level in 1:min(alg.NumMethods, integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] + 
                                              alg.AMatrices[stage - 2, 1, level] *
                                               integrator.k1[u_ind]
                end
            end
            for level in 1:min(alg.HighestEvalLevels[stage], integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] += alg.AMatrices[stage - 2, 2, level] *
                                               integrator.k_higher[u_ind]
                end
            end

            # "Remainder": Non-efficiently integrated
            for level in (alg.NumMethods + 1):(integrator.n_levels)
                @threaded for u_ind in integrator.level_u_indices_elements[level]
                    integrator.u_tmp[u_ind] = integrator.u[u_ind] + 
                                        alg.AMatrices[stage - 2, 1, alg.NumMethods] *
                                               integrator.k1[u_ind]
                end
            end
            if alg.HighestEvalLevels[stage] == alg.NumMethods
                for level in (alg.HighestEvalLevels[stage] + 1):(integrator.n_levels)
                    @threaded for u_ind in integrator.level_u_indices_elements[level]
                        integrator.u_tmp[u_ind] += alg.AMatrices[stage - 2, 2, alg.NumMethods] *
                                                   integrator.k_higher[u_ind]
                    end
                end
            end


            integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt

            # For statically non-uniform meshes/characteristic speeds
            integrator.coarsest_lvl = alg.HighestActiveLevels[stage]


            # Check if there are fewer integrators than grid levels (non-optimal method)
            if integrator.coarsest_lvl == alg.NumMethods
                # NOTE: This is supposedly more efficient than setting
                #integrator.coarsest_lvl = integrator.n_levels
                # and then using the level-dependent version

                integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

                @threaded for u_ind in eachindex(integrator.du)
                    integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
                end
            else
                
                # Joint RHS evaluation with all elements sharing this timestep
                
                
                integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, 
                            integrator.level_info_elements_acc[integrator.coarsest_lvl],
                            integrator.level_info_interfaces_acc[integrator.coarsest_lvl],
                            integrator.level_info_boundaries_acc[integrator.coarsest_lvl],
                            integrator.level_info_boundaries_orientation_acc[integrator.coarsest_lvl],
                            integrator.level_info_mortars_acc[integrator.coarsest_lvl],
                            integrator.coarsest_lvl)

                # Update k_higher of relevant levels
                for level in 1:integrator.coarsest_lvl
                    @threaded for u_ind in integrator.level_u_indices_elements[level]
                        integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
                    end
                end
            end
        end # end loop over different stages

        last_three_stages!(integrator, alg, prob.p)
    end # PERK4_Multi_EA step

    integrator.iter += 1
    integrator.t += integrator.dt

    @trixi_timeit timer() "Step-Callbacks" begin
        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
                return nothing
            end
        end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end
end

# get a cache where the RHS can be stored
get_du(integrator::PERK4_Multi_EA_Integrator) = integrator.du
get_tmp_cache(integrator::PERK4_Multi_EA_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::PERK4_Multi_EA_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::PERK4_Multi_EA_Integrator, dt)
    integrator.dt = dt
end

function get_proposed_dt(integrator::PERK4_Multi_EA_Integrator)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::PERK4_Multi_EA_Integrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PERK4_Multi_EA_Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)

    # TODO: Only for averaging callback (required for coupled Euler-acoustic simulations)
    resize!(integrator.uprev, new_size)
end
end # @muladd
