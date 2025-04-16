

using Alea 
using BenchmarkTools

function fun() 
    c = @alea begin
        a = uniform(DistIntOH{-200, 401},-100, 101)
        b = uniform(DistIntOH{-200, 401},-100, 101)
        c = uniform(DistIntOH{-200, 401},-100, 101)



        zero = DistIntOH{-200, 401}(0)


        type = 
            ifelse((a <= zero) | (b <= zero) | (c <= zero), 
                DistUInt{2}(3),
            ifelse(
                (a+b<=c) | (a+c<=b) | (b+c <= a), 
                DistUInt{2}(3),
            ifelse(
                prob_equals(a, b) & prob_equals(a, c) & prob_equals(b, c),
                DistUInt{2}(2), 
            ifelse(
                prob_equals(a, b) | prob_equals(a, c) | prob_equals(b, c), 
                DistUInt{2}(1), 
                DistUInt{2}(0)
            ))))

        type 

    end

    debug_info_ref = Ref{CuddDebugInfo}()
    pr(c, algo=Cudd(debug_info_ref=debug_info_ref))
    println("NUM_NODES_START")
    println(debug_info_ref[].num_nodes)
    println("NUM_NODES_END")
end 

fun()
