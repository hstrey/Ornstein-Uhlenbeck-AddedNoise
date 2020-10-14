using Turing, ReverseDiff, Memoization
using Statistics
using Random
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)
Turing.turnprogress(false)
using Distributions

@model g(rn,T) = begin
    sigma ~ Uniform(0.0,20.0)
    
    rn ~ MvNormal(zeros(T),sigma)
end

for i in 1:10
# create a sample from a normal
# sample from a Gaussian
    N = 1000
    d = Normal(0.0, i)
    data=rand(d,N)
    chn = sample(g(data,N),NUTS(0.65),1000)
    Turing.emptyrdcache()

    println("Std of chain: ",mean(Array(chn[:sigma])),"Std of data: ",std(data))
end
