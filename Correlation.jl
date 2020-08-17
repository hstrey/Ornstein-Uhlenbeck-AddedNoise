using Turing, ReverseDiff, Memoization
using DifferentialEquations
using Plots
using Statistics
using StatsPlots
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
using Distributions
using LinearAlgebra

# here we are simulating the signal of two coupled oscillators
# with amplitudes A1 and A2 of kBT/k and kBT/(k+c) where c is the coupling constant
# D = kBT/gamma
# in terms of the standard parameters of a OU process:
# sigma = sqrt(2*D)
# Theta = D/A

function corr_osc(c)
    # function that returns two time series that are correlated through a coupling coefficient c
    μ = 0.0 # mean is zero
    σ = sqrt(2) # D=1
    Θ1 = 1.0
    Θ2 = 1.0+abs(c)

    W1 = OrnsteinUhlenbeckProcess(Θ1,μ,σ,0.0,1.0)
    W2 = OrnsteinUhlenbeckProcess(Θ2,μ,σ,0.0,1.0)
    prob1 = NoiseProblem(W1,(0.0,100.0))
    prob2 = NoiseProblem(W2,(0.0,100.0))
    sol1 = solve(prob1;dt=0.1)
    sol2 = solve(prob2;dt=0.1)

    # creating the two correlated
    x1 = (sol1.u .+ sol2.u)/2
    if c>0 
        x2 = (sol1.u .- sol2.u)/2
    else
        x2 = (sol2.u .- sol1.u)/2
    end
    return x1,x2
end

# Ornstein-Uhlenbeck process
@model ou(rn,T,delta_t) = begin
    ampl ~ Uniform(0.0,5.0)
    b ~ beta(5.0,1.0)
    
    rn[1] ~ Normal(0,sqrt(ampl))
    
    for i=2:T
        rn[i] ~ Normal(rn[i-1]*b,sqrt(ampl*(1-b^2)))
    end
end

# Ornstein-Uhlenbeck process of two coupled oscillators
@model ou_corr(rn1,rn2,T,delta_t) = begin
    ampl1 ~ Uniform(0.0,5.0)
    ampl2 ~ Uniform(0.0,5.0)
    d ~ Uniform(0.0,5.0)
    b1 = exp(-delta_t*d/ampl1)
    b2 = exp(-delta_t*d/ampl2)

    rn1[1] ~ Normal(0,sqrt(ampl1))
    rn2[1] ~ Normal(0,sqrt(ampl2))   

    for i=2:T
        rn1[i] ~ Normal(rn1[i-1]*b1,sqrt(ampl1*(1-b1^2)))
        rn2[i] ~ Normal(rn2[i-1]*b2,sqrt(ampl2*(1-b2^2)))
    end
end

# Ornstein-Uhlenbeck process with added Gaussian noise
@model oupn(rn,T,delta_t,::Type{R}=Vector{Float64}) where {R} = begin
    ampl ~ Uniform(0.0,5.0)
    b ~ Beta(5.0,1.0)
    noise_ampl ~ Uniform(0.0,1)
    
    b = exp(-delta_t/tau)
    r = R(undef, T)
    
    r[1] ~ Normal(0,sqrt(ampl))
    
    for i=2:T
        r[i] ~ Normal(r[i-1]*b,sqrt(ampl*(1-b^2)))
    end
    rn ~ MvNormal(r,sqrt(noise_ampl))
end

x1, x2 = corr_osc(-0.8)

p1 = Plots.plot(x1)
p1 = Plots.plot!(x2)

pearson = Statistics.cor(x1,x2)
println(pearson)

# lets see whether we can estimate c from the data
y1 = x1 .+ x2
y2 = x1 .- x2
chn = sample(ou_corr(y1,y2,length(y1),0.1), NUTS(0.65), 10000)

print(describe(chn))
p2 = plot(chn)

ampl1 = Array(chn[:ampl1])
ampl2 = Array(chn[:ampl2])

if ampl1>ampl2
    c = (ampl1 .- ampl2)./ampl2
else
    c = (ampl1 .- ampl2)./ampl1
end

println("A1,A2: ",mean(ampl1),",",mean(ampl2))
println("c estimate: ",mean(c),"std: ",std(c))
