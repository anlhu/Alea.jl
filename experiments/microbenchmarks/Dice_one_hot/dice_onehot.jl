using Alea
using BenchmarkTools

# localARGS = ARGS
# num_bits = parse(Int64, localARGS[1])



function less(num_bits)
    nbpow = 2^num_bits
    # code = @alea begin
    a = discrete(DistUIntOH{nbpow}, [1/nbpow for _ in 1:nbpow])
    b = discrete(DistUIntOH{nbpow}, [1/nbpow for _ in 1:nbpow])
        # c = (a < b)
        # return c
    # end
    #~begin less
    return pr(a < b)
end

function equal(num_bits)
    nbpow = 2^num_bits

    a = discrete(DistUIntOH{nbpow}, [1/nbpow for _ in 1:nbpow])
    b = discrete(DistUIntOH{nbpow}, [1/nbpow for _ in 1:nbpow])

    #~begin less
    return pr(prob_equals(a, b))
end

function add(num_bits)
    nbpow = 2^num_bits

    a = discrete(DistUIntOH{2 * nbpow}, [1/nbpow for _ in 1:nbpow])
    b = discrete(DistUIntOH{2 * nbpow}, [1/nbpow for _ in 1:nbpow])

    #~begin less
    return expectation(a + b)
end
#~end

#~begin equals
# println(pr(prob_equals(a, b)))
#~end

#~begin sum
# println(expectation(a + b))

# for i = 1:15
#     t = @benchmark c = less($i)
#     med =  ((median(t).time)/10^9)
#     @show (i, med)
# end

# for i = 1:15
#     t = @benchmark c = equal($i)
#     med =  ((median(t).time)/10^9)
#     @show (i, med)
# end

for i = 7:7
    t = @benchmark c = add($i)
    med =  ((median(t).time)/10^9)
    @show (i, med)
end
