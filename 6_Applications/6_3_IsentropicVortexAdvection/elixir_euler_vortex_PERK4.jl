
using OrdinaryDiffEq, Plots
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

# define new structs inside a module to allow re-evaluating the file
module TrixiExtension
  using Trixi

  struct IndicatorVortex{Cache<:NamedTuple} <: Trixi.AbstractIndicator
    cache::Cache
  end

  function IndicatorVortex(semi)
    basis = semi.solver.basis
    alpha = Vector{real(basis)}()
    A = Array{real(basis), 2}
    indicator_threaded = [A(undef, nnodes(basis), nnodes(basis))
                          for _ in 1:Threads.nthreads()]
    cache = (; semi.mesh, alpha, indicator_threaded) # "Leading semicolon" makes this a named tuple

    return IndicatorVortex{typeof(cache)}(cache)
  end

  function (indicator_vortex::IndicatorVortex)(u::AbstractArray{<:Any,4},
                                              mesh, equations, dg, cache;
                                              t, kwargs...)
    mesh = indicator_vortex.cache.mesh
    alpha = indicator_vortex.cache.alpha
    indicator_threaded = indicator_vortex.cache.indicator_threaded
    resize!(alpha, nelements(dg, cache))


    # get analytical vortex center (based on assumption that center=[0.0,0.0]
    # at t=0.0 and that we stop after one period)
    domain_length = mesh.tree.length_level_0
    if t < 0.5 * domain_length
      center = (t, t)
    else
      center = (t-domain_length, t-domain_length)
    end

    Threads.@threads for element in eachelement(dg, cache)
      cell_id = cache.elements.cell_ids[element]
      coordinates = (mesh.tree.coordinates[1, cell_id], mesh.tree.coordinates[2, cell_id])
      # use the negative radius as indicator since the AMR controller increases
      # the level with increasing value of the indicator and we want to use
      # high levels near the vortex center
      alpha[element] = -periodic_distance_2d(coordinates, center, domain_length)
    end

    return alpha
  end

  function periodic_distance_2d(coordinates, center, domain_length)
    dx = @. abs(coordinates - center)
    dx_periodic = @. min(dx, domain_length - dx)
    return sqrt(sum(abs2, dx_periodic))
  end
end # module TrixiExtension

import .TrixiExtension

# Ratio of specific heats
gamma = 1.4

equations = CompressibleEulerEquations2D(gamma)

EdgeLength = 10

N_passes = 1
T_end = 2*EdgeLength * N_passes
tspan = (0.0, T_end)

"""
    initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)

The classical isentropic vortex test case as presented in 
https://spectrum.library.concordia.ca/id/eprint/985444/1/Paired-explicit-Runge-Kutta-schemes-for-stiff-sy_2019_Journal-of-Computation.pdf
"""
function initial_condition_isentropic_vortex(x, t, equations::CompressibleEulerEquations2D)
  # Evaluate error after full domain traversion
  if t == T_end
    t = 0
  end

  # initial center of the vortex
  inicenter = SVector(0.0, 0.0)
  # strength of the vortex
  S = 13.5
  # Radius of vortex
  R = 1.5
  # Free-stream Mach 
  M = 0.4
  # base flow
  v1 = 1.0
  v2 = 1.0
  vel = SVector(v1, v2)

  cent = inicenter + vel*t      # advection of center
  cent = x - cent               # distance to centerpoint
  cent = SVector(cent[2], -cent[1])
  r2 = cent[1]^2 + cent[2]^2

  f = (1 - r2) / (2 * R^2)

  rho = (1 - (S*M/pi)^2 * (gamma - 1)*exp(2*f) / 8)^(1/(gamma - 1))

  du = S/(2*π*R)*exp(f) # vel. perturbation
  vel = vel + du*cent
  v1, v2 = vel

  p = rho^gamma / (gamma * M^2)
  prim = SVector(rho, v1, v2, p)
  return prim2cons(prim, equations)
end
initial_condition = initial_condition_isentropic_vortex

surf_flux = flux_hllc # Better flux, allows much larger timesteps
PolyDeg = 3
solver = DGSEM(polydeg=PolyDeg, surface_flux=surf_flux)

coordinates_min = (-EdgeLength, -EdgeLength)
coordinates_max = ( EdgeLength,  EdgeLength)

Refinement = 6
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=Refinement,
                n_cells_max=100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = ())

amr_controller = ControllerThreeLevel(semi, TrixiExtension.IndicatorVortex(semi),
                                      base_level=Refinement,
                                      med_level=Refinement+1, med_threshold=-3.0,
                                      max_level=Refinement+3, max_threshold=-0.6)
                                     
amr_interval = 16 # PERK4 Multi
#amr_interval = 10 # PERK4 19

#amr_interval = 18 # NDBLSRK144
#amr_interval = 30 # DGLDDRK84_C
#amr_interval = 26 # RDPK3SpFSAL49
#amr_interval = 74 # RK4

amr_callback = AMRCallback(semi, amr_controller,
                           interval=amr_interval, 
                           adapt_initial_condition=true)

### AMR, Ref_Lvl = 6, Standard DGSEM ###

# E = 5, 7, 11, 19
CFL = 7.4

# PERK4 Standalone #
#CFL = 11.5 # S = 19

# OrdinaryDiffEq.jl methods

#CFL = 6.7 # NDBLSRK144
#CFL = 3.9 # DGLDDRK84_C
#CFL = 4.6 # RDPK3SpFSAL49
#CFL = 1.6 # RK4

stepsize_callback = StepsizeCallback(cfl = CFL)

callbacks = CallbackSet(summary_callback,
                        amr_callback,
                        stepsize_callback,
                        analysis_callback)                

###############################################################################
# run the simulation

dtRatios = [1, 0.5, 0.25, 0.125]
Stages = [19, 11, 7, 5]

cd(@__DIR__)
ode_algorithm = PERK4_Multi(Stages, "./", dtRatios)

#ode_algorithm = PERK4(19, "./")

@assert Threads.nthreads() == 1 "Provided data obtained on 1 thread"

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 42.0,
                  save_everystep=false, callback=callbacks);

summary_callback() # print the timer summary

###############################################################################

#=
ode_algorithm = NDBLSRK144(williamson_condition = false)
ode_algorithm = DGLDDRK84_C(williamson_condition = false)
ode_algorithm = RDPK3SpFSAL49()
ode_algorithm = RK4()

sol = solve(ode, ode_algorithm,
            dt = 42.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks, 
            adaptive = false);

summary_callback() # print the timer summary
=#