# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct PERK4_var_c{StageCallbacks}
    const NumStages::Int64
    stage_callbacks::StageCallbacks

    AMatrices::Matrix{Float64}
    AMatrix::Matrix{Float64}
    c::Vector{Float64}

    function PERK4_var_c(NumStages_::Int, BasePathMonCoeffs_::AbstractString,
                   stage_callbacks = ())
        newPERK4 = new{typeof(stage_callbacks)}(NumStages_, stage_callbacks)

        newPERK4.AMatrices, newPERK4.AMatrix, newPERK4.c = ComputePERK4_ButcherTableau(NumStages_,
                                                                                       BasePathMonCoeffs_)

        return newPERK4
    end
end # struct PERK4_var_c

function init(ode::ODEProblem, alg::PERK4_var_c;
              dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # PERK4 stages
    k1 = zero(u0)
    k_higher = zero(u0)
    
    t0 = first(ode.tspan)
    iter = 0

    integrator = PERK4_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                                  (prob = ode,), ode.f, alg,
                                  PERK_IntegratorOptions(callback, ode.tspan;
                                                         kwargs...), false,
                                  k1, k_higher, t0)

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
function solve(ode::ODEProblem, alg::PERK4_var_c;
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve_steps!(integrator)
end
end # @muladd
