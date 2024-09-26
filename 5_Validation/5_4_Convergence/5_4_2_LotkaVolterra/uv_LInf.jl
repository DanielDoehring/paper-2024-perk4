using NLsolve, OrdinaryDiffEq, LinearAlgebra, Plots

# See https://arxiv.org/pdf/2303.09317.pdf AN EXACT CLOSED-FORM SOLUTION OF THE LOTKA-VOLTERRA EQUATIONS

# Values from table for energy
h = 2.0

### Implicit function for xi_minus, xi_plus depending on h ###
function U(xi, h)
  [exp(xi[1]) - xi[1] - 1 - h]
end

function U_h(xi)
  return U(xi, h)
end

# Provided data from table for h = 0.3 (used as initial guess)

if h == 0.3
  xi_minus = -0.889
  xi_plus = 0.686
elseif h == 0.5
  xi_minus = -1.198
  xi_plus = 0.858
elseif h == 1.0
  xi_minus = -1.841
  xi_plus = 1.146
elseif h == 2.0 
  xi_minus = -2.948
  xi_plus = 1.505
end

### Solve for xi_minus, xi_plus (accurately) ###
#sol = nlsolve(U_h, [xi_plus], method = :trust_region, ftol = 0.0)
sol = nlsolve(U_h, [xi_plus], ftol = 0.0)
xi_plus = sol.zero[1]

#sol = nlsolve(U_h, [xi_minus], method = :trust_region, ftol = 0.0)
sol = nlsolve(U_h, [xi_minus], ftol = 0.0)
xi_minus = sol.zero[1]

### Compute tstar ###
tstar = 5.430320388771402

### Compute preys (u) and predators (v) populations from xi (analytical solution) ###
function uv(h, xi)
  u = h + 1 + xi + sqrt(abs((h + 1 + xi)^2 - exp(2*xi)))
  #u = h + 1 + xi + sqrt((h + 1 + xi)^2 - exp(2*xi))

  v = h + 1 + xi - sqrt(abs((h + 1 + xi)^2 - exp(2*xi)))
  #v = h + 1 + xi - sqrt((h + 1 + xi)^2 - exp(2*xi))

  return u, v
end

# If xi_desired = xi_plus
u, v = (h + 1 + xi_plus, h + 1 + xi_plus)
#u, v = uv(h, xi_plus)

# Compute IC 
u0, v0 = uv(h, xi_minus)
#u0, v0 = (h + 1 + xi_minus, h + 1 + xi_minus)

y0 = [u0; v0]

### Solve ODE ###

function LotkaVolterra(dy, y, p, t)
  dy[1] = y[1] * (1.0 - y[2])
  #dy[1] = y[1] - y[1]*y[2]

  dy[2] = y[2] * (y[1] - 1.0)
  #dy[2] = y[2]*y[1] - y[2]
end

tspan = (0.0, tstar)

p = 42.0
prob = ODEProblem(LotkaVolterra, y0, tspan, p)

N = 14

#########################################
### PERK 4 ###
#########################################

include("PERK4_Multi_2Level.jl")

Stages = [9, 5]
ode_algorithm = PERK4_Multi(Stages, "/home/daniel/git/Paper_PERK4/Data/LotkaVolterra/")

errors = zeros(N)

println("Component-wise errors:")

for i = 0:N-1
  dt = 1.0/(2.0^i)
  #println("dt: ", dt)
  sol_ode = solve_(prob, ode_algorithm, dt = dt);

  errors[i+1] = norm(sol_ode.u[end] - [u, v], Inf)
  println(errors[i+1])
end