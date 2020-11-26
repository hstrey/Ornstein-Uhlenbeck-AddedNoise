using Turing, ReverseDiff, Memoization
using DifferentialEquations
using Plots
using Statistics
using StatsPlots
using DataFrames
using CSV
using NLsolve
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
Turing.turnprogress(true)
using Distributions
using LinearAlgebra

# here we are simulating the signal of two coupled oscillators
# with amplitudes A1 and A2 of kBT/k and kBT/(k+c) where c is the coupling constant
# D = kBT/gamma
# in terms of the standard parameters of a OU process:
# sigma = sqrt(2*D)
# Theta = D/A

function corr_osc(c,N,delta_t)
    # function that returns two time series that are correlated through a coupling coefficient c
    μ = 0.0 # mean is zero
    σ = sqrt(2) # D=1
    Θ1 = 1.0
    Θ2 = 1.0+abs(c)
    total_time = N*delta_t
    W1 = OrnsteinUhlenbeckProcess(Θ1,μ,σ,0.0,1.0)
    W2 = OrnsteinUhlenbeckProcess(Θ2,μ,σ,0.0,1.0)
    prob1 = NoiseProblem(W1,(0.0,total_time))
    prob2 = NoiseProblem(W2,(0.0,total_time))
    sol1 = solve(prob1;dt=delta_t)
    sol2 = solve(prob2;dt=delta_t)

    # creating the two correlated
    x1 = (sol1.u .+ sol2.u)/2
    if c>0 
        x2 = (sol1.u .- sol2.u)/2
    else
        x2 = (sol2.u .- sol1.u)/2
    end
    return x1,x2
end

function calc_fundstats(x)
    return x[1]^2+x[end]^2,sum(x[2:end-1].^2),sum(x[1:end-2].*x[2:end-1])
end

function phi_deriv(F,x,a1ep,a1ss,a1c,a2ep,a2ss,a2c,delta_t,N)
    # x[1] = A1, x[2] = A2, x[3]=D
    A1 = x[1]
    A2 = x[2]
    D = x[3]
    b1 = b(D,A1,delta_t)
    b2 = b(D,A2,delta_t)
    Q1 = q(a1ep,a1ss,a1c,b1)
    Q2 = q(a2ep,a2ss,a2c,b2)
    dQ1 = dqdB(a1ep,a1ss,a1c,b1)
    dQ2 = dqdB(a2ep,a2ss,a2c,b2)
    F[1] = -N*A1^2/2 + A1*Q1/2 + b1*D*delta_t*(A1*b1*(N-1)/(1-b1^2)-dQ1/2)
    F[2] = -N*A2^2/2 + A2*Q2/2 + b2*D*delta_t*(A2*b2*(N-1)/(1-b2^2)-dQ2/2)
    F[3] = (b1*(N-1)/(1-b1^2)-dQ1/A1/2)*b1/A1 + (b2*(N-1)/(1-b2^2)-dQ2/A2/2)*b2/A2
end

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


# here we would like to run MCMC on several artificial data sets with varying correlation coefficients
# we should test rho from -0.9 to 0.9

results = DataFrame(rho = Float64[],
                    coupling = Float64[],
                    pearson = Float64[],
                    a1 = Float64[],
                    da1 = Float64[],
                    a2 = Float64[],
                    da2 = Float64[],
                    d = Float64[],
                    dd = Float64[],
                    da1da2 = Float64[],
                    da1dd = Float64[],
                    dasdd = Float64[],
                    a1ep = Float64[],
                    a1ss = Float64[],
                    a1c = Float64[],
                    a2ep = Float64[],
                    a2ss = Float64[],
                    a2c = Float64[],
                    c = Float64[],
                    dc = Float64[])

tries = 500
rho = 0.5
delta_t = 0.3
N = 1000
coupling = 2*abs(rho)/(1-abs(rho))*sign(rho)
println("coupling: ",coupling," rho: ",rho)
# do rho tries times
for i in 1:tries
     x1, x2 = corr_osc(coupling,N,delta_t)
    pearson = Statistics.cor(x1,x2)
    println("i = ",i,"Pearson: ",pearson)

    # lets see whether we can estimate c from the data
    y1 = x1 .+ x2
    y2 = x1 .- x2

    a1ep,a1ss,a1c = calc_fundstats(y1)
    a2ep,a2ss,a2c = calc_fundstats(y2)

    # Ornstein-Uhlenbeck process of two coupled oscillators
    model_fct = ou_corr(y1,y2,length(y1),delta_t)
    chn = sample(model_fct, NUTS(0.65), 5000)
#        Turing.emptyrdcache()

    print(describe(chn))
    println(chn)
    ampl1 = Array(chn[:ampl1])
    ampl2 = Array(chn[:ampl2])
    d = Array(chn[:d])

    a1 = mean(ampl1)
    a2 = mean(ampl2)
    da1 = std(ampl1)
    da2 = std(ampl2)
    da1da2 = cov(ampl1,ampl2)
    da1dd = cov(ampl1,d)
    da2dd = cov(ampl2,d)
    d_mean = mean(d)
    dd = std(d)

    if a1>a2
        c = (ampl1 .- ampl2)./ampl2
    else
        c = (ampl1 .- ampl2)./ampl1
    end
    c_mean = mean(c)
    dc = std(c)

    println("A1,A2: ",a1,",",a2)
    println("c estimate: ",c_mean,"std: ",dc)
    println("cross corr: ",da1da2,da1dd,da2dd)

    push!(results,[rho,coupling,pearson,a1,da1,a2,da2,d_mean,dd,da1da2[1],da1dd[1],da2dd[1],a1ep,a1ss,a1c,a2ep,a2ss,a2c,c_mean,dc])
end

println(results)
CSV.write("correlations1k05.csv",results)

