using Distributions, Random

# function to produce N points of a OU process with variance A, relaxation time τ at interval Δt
function ou(A,τ,N,Δt)
    B = exp(-Δt/τ)
    x = [rand(Normal(0,sqrt(A)))]
    for i in 2:N
        push!(x,rand(Normal(x[end]*B,sqrt(A*(1-B^2)))))
    end
    return x
end

    
