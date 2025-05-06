using Alea
using Test

function Base.isapprox(d1::AbstractDict, d2::AbstractDict)
    issetequal(keys(d1), keys(d2)) && all(isapprox(d1[k], d2[k],rtol=0.01) for k in keys(d1))
end

# Basics --------------------------------- 
code = @alea begin 
            # This is where you write your probabilistic program
            if flip(0.3) flip(0.5) else false end
end

pr(code)

# Composition -----------------------------------------

code = @alea begin
            x = flip(0.3)
            y = if x flip(0.7) else flip(0.3) end
            z = x
            y & z
end
pr(code)

# Observations ---------------------------------------

code = @alea begin
            x = flip(0.5)
            y = flip(0.5)
            observe(x | y)
            x
end

pr(code)


# Functions ----------------------------------

code = @alea begin
            f(x) = if x flip(0.7) else flip(0.3) end
            f(if flip(0.3) flip(0.7) else false end)
end 

pr(code)

# Network Problem ------------------------------------

code = @alea begin
            s1 = true
            route = flip(0.5)
            s2 = if route s1 else false end
            s3 = if route false else s1 end
            
            drop_s2 = flip(0.01)
            drop_s3 = flip(0.0001)
            s4 = (s2 & !drop_s2) | (s3 & !drop_s3)
            
            observe(!s4)
            route
end

pr(code)