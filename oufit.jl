using Turing
using ReverseDiff
using ADTypes
using Plots

include("./ou.jl")

# Ornstein-Uhlenbeck process
@model ou_noise(rn,T,delta_t,::Type{R}=Vector{Float64}) where {R} = begin
    ampl ~ Uniform(0.0,2.0)
    τ ~ Uniform(0.5,1.5)
    b = exp(-delta_t/τ)
    noise_ampl ~ Uniform(0.0,2.0)

    r = R(undef, T)
    r[1] ~ Normal(0,ampl)
    for i=2:T
        r[i] ~ Normal(r[i-1]*b,ampl*sqrt(1-b^2))
    end
    rn ~ MvNormal(r,noise_ampl)
end

# Ornstein-Uhlenbeck process
@model ou_mnoise(rn,T,delta_t,::Type{R}=Vector{Float64}) where {R} = begin
    ampl ~ Uniform(0.0,2.0)
    τ ~ Uniform(0.5,1.5)
    b = exp(-delta_t/τ)
    noise_ampl ~ Uniform(0.0,2.0)
    mn_ampl ~ Uniform(0.0,1.0)

    r = R(undef, T)
    r[1] ~ Normal(0,ampl)
    for i=2:T
        r[i] ~ Normal(r[i-1]*b,ampl*sqrt(1-b^2))
    end
    rn ~ MvNormal(r,sqrt.(nm_ampl^2 .* r .^2) .+ noise_ampl)
end

x = ou(1.0,1.0,1000,0.1)
xn = x .+ rand(Normal(0,0.5),length(x))

modeloun = ou_noise(xn,length(xn),0.1)
chn = sample(modeloun, NUTS(0.65, adtype=AutoReverseDiff()), 3000)

advi = ADVI(20, 1000, AutoReverseDiff())
q = vi(modeloun, advi)
q_sample = rand(q, 5000)

mean(q_sample[1,:]), std(q_sample[1,:])
mean(q_sample[2,:]), std(q_sample[2,:])
mean(q_sample[3,:]), std(q_sample[3,:])
