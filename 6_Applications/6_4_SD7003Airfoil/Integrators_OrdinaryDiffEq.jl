using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

U_inf = 0.2
c_inf = 1.0

rho_inf = 1.4 # with gamma = 1.4 => p_inf = 1.0

Re = 10000.0
airfoil_cord_length = 1.0

t_c = airfoil_cord_length / U_inf

aoa = 4 * pi/180
u_x = U_inf * cos(aoa)
u_y = U_inf * sin(aoa)

gamma = 1.4
prandtl_number() = 0.72
mu() = rho_inf * U_inf * airfoil_cord_length / Re

equations = CompressibleEulerEquations2D(gamma)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesPrimitive())

@inline function initial_condition_mach02_flow(x, t, equations)
  # set the freestream flow parameters
  rho_freestream = 1.4

  v1 = 0.19951281005196486 # 0.2 * cos(aoa)
  v2 = 0.01395129474882506 # 0.2 * sin(aoa)
  
  p_freestream = 1.0

  prim = SVector(rho_freestream, v1, v2, p_freestream)
  return prim2cons(prim, equations)
end

initial_condition = initial_condition_mach02_flow

# Boundary conditions for free-stream testing
boundary_condition_free_stream = BoundaryConditionDirichlet(initial_condition)

velocity_bc_airfoil = NoSlip((x, t, equations) -> SVector(0.0, 0.0))
heat_bc = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_airfoil = BoundaryConditionNavierStokesWall(velocity_bc_airfoil, heat_bc)

polydeg = 3

surf_flux = flux_hllc
vol_flux = flux_chandrashekar
solver = DGSEM(polydeg = polydeg, surface_flux = surf_flux,
               volume_integral = VolumeIntegralFluxDifferencing(vol_flux))


###############################################################################
# Get the uncurved mesh from a file (downloads the file if not available locally)

cd(@__DIR__)
path = "./"
mesh_file = path * "sd7003_laminar_straight_sided_Trixi.inp"

boundary_symbols = [:Airfoil, :FarField]

mesh = P4estMesh{2}(mesh_file, polydeg = polydeg, boundary_symbols = boundary_symbols,
                    initial_refinement_level = 0)

# t = 30 t_c
#restart_file = "restart_NDBLSRK144.h5"
#restart_file = "restart_TD.h5" 
#restart_file = "restart_CKL.h5"
#restart_file = "restart_RK4.h5"
#restart_filename = joinpath("out", restart_file)


boundary_conditions = Dict(:FarField => boundary_condition_free_stream,
                           :Airfoil => boundary_condition_slip_wall)

boundary_conditions_parabolic = Dict(:FarField => boundary_condition_free_stream,
                                     :Airfoil => boundary_condition_airfoil)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions = (boundary_conditions,
                                                                    boundary_conditions_parabolic))

###############################################################################
# ODE solvers, callbacks etc.


tspan = (0.0, 30 * t_c) # Try to get into a state where initial pressure wave is gone
ode = semidiscretize(semi, tspan) # for OrdinaryDiffEq integrators

# Timespan for measurements over 5 * t_c
#tspan = (load_time(restart_filename), 35 * t_c)
#ode = semidiscretize(semi, tspan, restart_filename) # for OrdinaryDiffEq integrators

summary_callback = SummaryCallback()

# Choose analysis interval such that roughly every dt_c = 0.005 a record is taken
analysis_interval = 20 # NDBLSRK144
#analysis_interval = 33 # DGLDDRK84_C
#analysis_interval = 48 # CKLLSRK95_4S
#analysis_interval = 88 # RK4

f_aoa() = aoa
f_rho_inf() = rho_inf
f_U_inf() = U_inf
f_linf() = airfoil_cord_length

drag_coefficient = AnalysisSurfaceIntegral(semi, :Airfoil,
                                           DragCoefficientPressure(f_aoa(), f_rho_inf(),
                                                                   f_U_inf(), f_linf()))

drag_coefficient_shear_force = AnalysisSurfaceIntegral(semi, :Airfoil,
                                                       DragCoefficientShearStress(f_aoa(),
                                                                                  f_rho_inf(),
                                                                                  f_U_inf(),
                                                                                  f_linf()))

lift_coefficient = AnalysisSurfaceIntegral(semi, :Airfoil,
                                           LiftCoefficientPressure(f_aoa(), f_rho_inf(),
                                                                   f_U_inf(), f_linf()))

analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     output_directory = "out",
                                     save_analysis = true,
                                     analysis_errors = Symbol[],
                                     analysis_integrals = (drag_coefficient,
                                                           drag_coefficient_shear_force,
                                                           lift_coefficient))

stepsize_callback = StepsizeCallback(cfl = 7.7) # NDBLSRK144
#stepsize_callback = StepsizeCallback(cfl = 5.5) # DGLDDRK84_C
#stepsize_callback = StepsizeCallback(cfl = 3.2) # CKLLSRK95_4S
#stepsize_callback = StepsizeCallback(cfl = 1.9) # RK4

# For plots etc
save_solution = SaveSolutionCallback(interval = 2000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     output_directory="out")

alive_callback = AliveCallback(alive_interval = 200)

save_restart = SaveRestartCallback(interval = Int(10^7), # Only at end
                                   save_final_restart = true)

callbacks = CallbackSet(#analysis_callback, # For measurements
                        stepsize_callback, # For measurements: Fixed timestep (do not use this)
                        alive_callback, # Not needed for measurement run
                        #save_solution, # For plotting during measurement run
                        save_restart, # For restart with measurements
                        summary_callback);

###############################################################################
# run the simulation

ode_algorithm = NDBLSRK144(williamson_condition = false, thread = OrdinaryDiffEq.True())
#ode_algorithm = DGLDDRK84_C(williamson_condition = false, thread = OrdinaryDiffEq.True())
#ode_algorithm = CKLLSRK95_4S(thread = OrdinaryDiffEq.True())
#ode_algorithm = RK4(thread = OrdinaryDiffEq.True())

dt = 1.2e-3 # NDBLSRK144
#dt = 7.4e-4 # DGLDDRK84_C
#dt = 5.1e-4 # CKLLSRK95_4S
#dt = 3e-4 # RK4

@assert Threads.nthreads() == 24 "Provided data obtained on 24 threads"

sol = solve(ode, ode_algorithm,
            dt = dt, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks, 
            adaptive = false);

summary_callback() # print the timer summary
