using Alea 
using BenchmarkTools 

function binomial(n::DistUInt{W}, p, max::Int) where W 
    output = DistUInt{W}(0)
    for i in max-1:-1:0 
        output += ifelse((DistUInt{W}(i) < n) & flip(p), 
					                  DistUInt{W}(1), DistUInt{W}(0))
    end 
    return output 
end 

function fun()
    DInt = DistUInt{12}
    param = uniform(DInt, 0, 401) + DInt(100)
    nummet = binomial(param, 0.5, 500) + DInt(1)
    numinfected = binomial(nummet, 0.3, 501)
    pr(numinfected)
end

x = @benchmark fun() 

println((median(x).time)/10^9)
