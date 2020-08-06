using DifferentialEquations
using Plots
using Statistics

# here we are simulating the signal of two coupled oscillators
# with amplitudes A1 and A2 of kBT/k and kBT/(k+c) where c is the coupling constant
# D = kBT/gamma
# in terms of the standard parameters of a OU process:
# sigma = sqrt(2*D)
# Theta = D/A

function corr_osc(c)

μ = 0.0 # mean is zero
σ = sqrt(2) # D=1
Θ1 = 1.0
Θ2 = 1.0+c

W1 = OrnsteinUhlenbeckProcess(Θ1,μ,σ,0.0,1.0)
W2 = OrnsteinUhlenbeckProcess(Θ2,μ,σ,0.0,1.0)
prob1 = NoiseProblem(W1,(0.0,100.0))
prob2 = NoiseProblem(W2,(0.0,100.0))
sol1 = solve(prob1;dt=0.1)
sol2 = solve(prob2;dt=0.1)

# creating the two correlated 
x1 = sol1.u .+ sol2.u
x2 = sol1.u .- sol2.u
return x1,x2

end

x1, x2 = corr_osc(5)

Plots.plot(x1)
Plots.plot!(x2)

println(Statistics.cor(x1,x2))
