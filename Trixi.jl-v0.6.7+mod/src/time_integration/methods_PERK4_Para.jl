# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct PERK4_Para{StageCallbacks}
    const NumStages::Int64
    stage_callbacks::StageCallbacks

    AMatrices::Matrix{Float64}
    AMatrix::Matrix{Float64}
    c::Vector{Float64}

    function PERK4_Para(NumStages_::Int, BasePathMonCoeffs_::AbstractString,
                   stage_callbacks = ())
        newPERK4_Para = new{typeof(stage_callbacks)}(NumStages_, stage_callbacks)

        newPERK4_Para.AMatrices, newPERK4_Para.AMatrix, newPERK4_Para.c = ComputePERK4_ButcherTableau(NumStages_,
                                                                                       BasePathMonCoeffs_)

        return newPERK4_Para
    end
end # struct PERK4_Para

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK4_Para_Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
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
    # PERK4_Para stages:
    k1::uType
    k_higher::uType
    t_stage::RealT

    # TODO: Not best solution since this is not needed for hyperbolic problems
    du_ode_hyp::uType 
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PERK4_Para_Integrator, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

function init(ode::ODEProblem, alg::PERK4_Para;
              dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # PERK4_Para stages
    k1 = zero(u0)
    k_higher = zero(u0)

    du_ode_hyp = zero(u0) # TODO: Not best solution since this is not needed for hyperbolic problems

    t0 = first(ode.tspan)
    iter = 0

    integrator = PERK4_Para_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                                  (prob = ode,), ode.f, alg,
                                  PERK_IntegratorOptions(callback, ode.tspan;
                                                         kwargs...), false,
                                  k1, k_higher, t0,
                                  du_ode_hyp)

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
function solve(ode::ODEProblem, alg::PERK4_Para;
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve_steps!(integrator)
end

function solve_steps!(integrator::PERK4_Para_Integrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        step!(integrator)
    end # "main loop" timer

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

function k1!(integrator::PERK4_Para_Integrator, p, c)
    integrator.f(integrator.du, integrator.u, p, integrator.t, integrator.du_ode_hyp)

    @threaded for i in eachindex(integrator.du)
        integrator.k1[i] = integrator.du[i] * integrator.dt
    end

    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] + c[2] * integrator.k1[i]
    end

    # TODO: Move away from here, not really belonging to stage 1!
    integrator.t_stage = integrator.t + c[2] * integrator.dt
end

function last_three_stages!(integrator::PERK4_Para_Integrator, alg, p)
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

        integrator.f(integrator.du, integrator.u_tmp, p, integrator.t_stage, integrator.du_ode_hyp)

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

    integrator.f(integrator.du, integrator.u_tmp, p, integrator.t + alg.c[alg.NumStages] * integrator.dt, integrator.du_ode_hyp)
    
    @threaded for u_ind in eachindex(integrator.u)
        # "Own" PairedExplicitRK based on SSPRK33.
        # Note that 'k_higher' carries the values of K_{S-1}
        # and that we construct 'K_S' "in-place" from 'integrator.du'
        integrator.u[u_ind] += 0.5 * (integrator.k_higher[u_ind] + integrator.du[u_ind] * integrator.dt)
    end
end

function step!(integrator::PERK4_Para_Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
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

        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)

        @threaded for u_ind in eachindex(integrator.du)
            integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
        end

        for stage in 3:(alg.NumStages - 3)
            # Construct current state
            @threaded for u_ind in eachindex(integrator.u)
                integrator.u_tmp[u_ind] = integrator.u[u_ind] +
                                          alg.AMatrices[stage - 2, 1] *
                                          integrator.k1[u_ind] +
                                          alg.AMatrices[stage - 2, 2] *
                                          integrator.k_higher[u_ind]
            end

            integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt

            integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage, integrator.du_ode_hyp)

            @threaded for i in eachindex(integrator.du)
                integrator.k_higher[i] = integrator.du[i] * integrator.dt
            end
        end

        last_three_stages!(integrator, alg, prob.p)
    end # PERK4_Para step

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
get_du(integrator::PERK4_Para_Integrator) = integrator.du
get_tmp_cache(integrator::PERK4_Para_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::PERK4_Para_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::PERK4_Para_Integrator, dt)
    integrator.dt = dt
end

function get_proposed_dt(integrator::PERK4_Para_Integrator)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::PERK4_Para_Integrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PERK4_Para_Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)

    resize!(integrator.du_ode_hyp, new_size)
end
end # @muladd
