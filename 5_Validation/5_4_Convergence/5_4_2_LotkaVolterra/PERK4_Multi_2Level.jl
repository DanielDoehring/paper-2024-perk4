using OrdinaryDiffEq
using SimpleUnPack: @unpack

function read_file(FilePath::AbstractString, DataType::Type=Float64)
  @assert isfile(FilePath) "Couldn't find file"
  Data = zeros(DataType, 0)
  open(FilePath, "r") do File
    while !eof(File)     
      LineContent = readline(File)     
      append!(Data, parse(DataType, LineContent))
    end
  end
  NumLines = length(Data)

  return NumLines, Data
end

function ComputePERK4_Multi_ButcherTableau(Stages::Vector{Int64}, NumStages::Int, BasePathMonCoeffs::AbstractString)
                                     
  # Use linear increasing timesteps for free timesteps
  c = zeros(NumStages)
  for k in 2:NumStages-4
    c[k] = (k - 1)/(NumStages - 4) # Equidistant timestep distribution (similar to PERK2)
  end
  
  # Current approach: Use ones for simplicity
  #=
  c = ones(NumStages)
  c[1] = 0.0
  =#

  c[NumStages - 3] = 1.0
  c[NumStages - 2] = 0.479274057836310
  c[NumStages - 1] = sqrt(3)/6 + 0.5
  c[NumStages]     = -sqrt(3)/6 + 0.5
  
  println("Timestep-split: "); display(c); println("\n")

  # For the p = 4 method there are less free coefficients
  CoeffsMax = NumStages - 5

  AMatrices = zeros(length(Stages), CoeffsMax, 2)
  for i = 1:length(Stages)
    AMatrices[i, :, 1] = c[3:NumStages-3]
  end

  # Datastructure indicating at which stage which level is evaluated
  ActiveLevels = [Vector{Int64}() for _ in 1:NumStages]
  # k1 is evaluated at all levels
  ActiveLevels[1] = 1:length(Stages)

  # Datastructure indicating at which stage which level contributes to state
  EvalLevels = [Vector{Int64}() for _ in 1:NumStages]
  # k1 is evaluated at all levels
  EvalLevels[1] = 1:length(Stages)
  # Second stage: Only finest method
  EvalLevels[2] = [1]

  for level in eachindex(Stages)

    NumStageEvals = Stages[level]
    PathMonCoeffs = BasePathMonCoeffs * "a_" * string(NumStageEvals) * "_" * string(NumStages) * ".txt"
    #PathMonCoeffs = BasePathMonCoeffs * "a_" * string(NumStageEvals) * ".txt"
    NumMonCoeffs, A = read_file(PathMonCoeffs, Float64)
    @assert NumMonCoeffs == NumStageEvals - 5

    if NumMonCoeffs > 0
      AMatrices[level, CoeffsMax - NumMonCoeffs + 1:end, 1] -= A
      AMatrices[level, CoeffsMax - NumMonCoeffs + 1:end, 2]  = A
    end

    # Add active levels to stages
    for stage = NumStages:-1:NumStages-(3 + NumMonCoeffs)
      push!(ActiveLevels[stage], level)
    end

    # Add eval levels to stages
    for stage = NumStages:-1:NumStages-(3 + NumMonCoeffs) - 1
      push!(EvalLevels[stage], level)
    end
  end
  # Shared matrix
  AMatrix = [0.364422246578869 0.114851811257441
             0.1397682537005989 0.648906880894214
             0.1830127018922191 0.028312163512968]

  HighestActiveLevels = maximum.(ActiveLevels)
  HighestEvalLevels   = maximum.(EvalLevels)

  for i = 1:length(Stages)
    println("A-Matrix of Butcher tableau of level " * string(i))
    display(AMatrices[i, :, :]); println()
  end

  println("\nActive Levels:"); display(ActiveLevels); println()
  println("\nHighestEvalLevels:"); display(HighestEvalLevels); println()

  return AMatrices, AMatrix, c, ActiveLevels, HighestActiveLevels, HighestEvalLevels
end

mutable struct PERK4_Multi
  const NumStageEvalsMin::Int64
  const NumMethods::Int64
  const NumStages::Int64

  AMatrices::Array{Float64, 3}
  AMatrix::Matrix{Float64}
  c::Vector{Float64}
  ActiveLevels::Vector{Vector{Int64}}
  HighestActiveLevels::Vector{Int64}
  HighestEvalLevels::Vector{Int64}

  function PERK4_Multi(Stages_::Vector{Int64},
                       BasePathMonCoeffs_::AbstractString)

    newPERK4_Multi = new(minimum(Stages_),
                          length(Stages_),
                          maximum(Stages_))

    newPERK4_Multi.AMatrices, newPERK4_Multi.AMatrix, newPERK4_Multi.c, 
    newPERK4_Multi.ActiveLevels, newPERK4_Multi.HighestActiveLevels, newPERK4_Multi.HighestEvalLevels = 
      ComputePERK4_Multi_ButcherTableau(Stages_, newPERK4_Multi.NumStages, BasePathMonCoeffs_)

    return newPERK4_Multi
  end
end # struct PERK4_Multi

# Wrapper type for solutions from Trixi.jl's own time integrators, partially mimicking
# SciMLBase.ODESolution
struct TimeIntegratorSolution{tType, uType, P}
  t::tType
  u::uType
  prob::P
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct PERK_IntegratorOptions{Callback}
  callback::Callback # callbacks; used in Trixi
  adaptive::Bool # whether the algorithm is adaptive; ignored
  dtmax::Float64 # ignored
  maxiters::Int # maximal numer of time steps
  tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function PERK_IntegratorOptions(callback, tspan; maxiters=typemax(Int), kwargs...)
  PERK_IntegratorOptions{typeof(callback)}(callback, false, Inf, maxiters, [last(tspan)])
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK4_Multi_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, PERK_IntegratorOptions}
  u::uType
  du::uType
  u_tmp::uType
  t::RealT
  dt::RealT # current time step
  dtcache::RealT # ignored
  iter::Int # current number of time steps (iteration)
  p::Params # will be the semidiscretization from Trixi
  sol::Sol # faked
  f::F
  alg::Alg # This is our own class written above; Abbreviation for ALGorithm
  opts::PERK_IntegratorOptions
  finalstep::Bool # added for convenience
  # PERK4_Multi stages:
  k1::uType
  k_higher::uType
  k_S1::uType # Required for third & fourth order
  level_u_indices_elements::Vector{Vector{Int64}}
  t_stage::RealT
  coarsest_lvl::Int64
  n_levels::Int64
end


# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PERK4_Multi_Integrator, field::Symbol)
  if field === :stats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end


# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve_(ode::ODEProblem, alg::PERK4_Multi; dt, callback=nothing, kwargs...)

  u0    = copy(ode.u0)
  du    = zero(u0) # previously: similar(u0)
  u_tmp = zero(u0)

  # PERK4_Multi stages
  k1       = zero(u0)
  k_higher = zero(u0)
  k_S1     = zero(u0)

  t0 = first(ode.tspan)
  iter = 0

  ### Done with setting up for handling of level-dependent integration ###

  level_u_indices_elements = [Vector{Int64}() for _ in 1:2]
  level_u_indices_elements[1] = [1]
  level_u_indices_elements[2] = [2]

  integrator = PERK4_Multi_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                                      (prob=ode,), ode.f, alg,
                                      PERK_IntegratorOptions(callback, ode.tspan; kwargs...), false,
                                      k1, k_higher, k_S1, 
                                      level_u_indices_elements,
                                      t0, -1, 2)

  # Start actual solve
  solve_!(integrator)
end


function solve_!(integrator::PERK4_Multi_Integrator)
  @unpack prob = integrator.sol
  @unpack alg = integrator
  t_end = last(prob.tspan)
  callbacks = integrator.opts.callback

  integrator.finalstep = false

  while !integrator.finalstep
    if isnan(integrator.dt)
      error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end || isapprox(integrator.t + integrator.dt, t_end)
      integrator.dt = t_end - integrator.t
      terminate!(integrator)
    end

    # k1: Evaluated on entire domain / all levels
    integrator.f(integrator.du, integrator.u, prob.p, integrator.t)

    for i in eachindex(integrator.du)
      integrator.k1[i] = integrator.du[i] * integrator.dt
    end

    integrator.t_stage = integrator.t + alg.c[2] * integrator.dt
    # k2: Here always evaluated for finest scheme (Allow currently only max. stage evaluations)
    for i in eachindex(integrator.u)
      integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)
    
    # Update finest level only
    for u_ind in integrator.level_u_indices_elements[1]
      integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
    end

    for stage = 3:alg.NumStages - 3
      # Construct current state
      for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i]
      end

      # Loop over different methods with own associated level
      for level = 1:min(alg.NumMethods, integrator.n_levels)
        for u_ind in integrator.level_u_indices_elements[level]
          integrator.u_tmp[u_ind] += alg.AMatrices[level, stage - 2, 1] * integrator.k1[u_ind]
        end
      end
      for level = 1:min(alg.HighestEvalLevels[stage], integrator.n_levels)
        for u_ind in integrator.level_u_indices_elements[level]
          integrator.u_tmp[u_ind] += alg.AMatrices[level, stage - 2, 2] * integrator.k_higher[u_ind]
        end
      end

      integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt

      integrator.coarsest_lvl = min(alg.HighestActiveLevels[stage], integrator.n_levels)
      
      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)
      
      # Update k_higher of relevant levels
      for level in 1:integrator.coarsest_lvl          
        for u_ind in integrator.level_u_indices_elements[level]
          integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
        end
      end
    end

    # Last three stages: Same Butcher Matrix
    for stage = 1:3
      for u_ind in eachindex(integrator.u)
        integrator.u_tmp[u_ind] = integrator.u[u_ind] + alg.AMatrix[stage, 1] * integrator.k1[u_ind] + 
                                                        alg.AMatrix[stage, 2] * integrator.k_higher[u_ind]
      end
      integrator.t_stage = integrator.t + alg.c[alg.NumStages - 3 + stage] * integrator.dt

      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

      for u_ind in eachindex(integrator.u)
        integrator.k_higher[u_ind] = integrator.du[u_ind] * integrator.dt
      end

      if stage == 2
        for u_ind in eachindex(integrator.u)
          integrator.k_S1[u_ind] = integrator.k_higher[u_ind]
        end
      end
    end

    for u_ind in eachindex(integrator.u)
      integrator.u[u_ind] += 0.5 * (integrator.k_S1[u_ind] + integrator.k_higher[u_ind])
    end

    integrator.iter += 1
    integrator.t += integrator.dt

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
      @warn "Interrupted. Larger maxiters is needed."
      terminate!(integrator)
    end
  end # "main loop" timer
  
  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u),
                                integrator.sol.prob)
end

# stop the time integration
function terminate!(integrator::PERK4_Multi_Integrator)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end