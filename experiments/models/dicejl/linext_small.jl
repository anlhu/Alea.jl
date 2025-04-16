using Alea
using BenchmarkTools

function fun()

    c = @alea begin 

        uniq_count = 4
        
        bits = Int(floor(log2(uniq_count)) + 4)
        partial = [[1, 3], [1, 4]]

        ranks = [uniform(DistInt{bits}, 0, uniq_count) for i in 1:uniq_count]
        for i in 1:uniq_count
            for j in i+1:uniq_count
                observe(!prob_equals(ranks[i], ranks[j]))
            end
        end
        

        for comp in partial 
            observe(ranks[comp[1]] < ranks[comp[2]])
        end

        (ranks[1], ranks[2], ranks[3], ranks[4])
    end
    pr(c)
end 
x = @benchmark fun()

println((median(x).time)/10^9)