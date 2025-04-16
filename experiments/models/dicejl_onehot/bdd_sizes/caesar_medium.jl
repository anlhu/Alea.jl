using Alea
using BenchmarkTools

function fun()
    c = @alea begin 
        char_freqs = [0.082, 0.018, 0.024, 0.04, 0.123, 0.022, 0.022, 0.066, 0.069, 0.001, 0.008, 0.045, 0.024, 0.069, 0.073, 0.018, 0.002, 0.055, 0.067, 0.092, 0.028, 0.009, 0.023, 0.001, 0.018, 0.001]
        key = uniform(DistUIntOH{26}, 0, 26)
        
        function sendChar(key::DistUIntOH{26}, c::DistUIntOH{26})
            gen = discrete(DistUIntOH{26}, char_freqs)
            enc = key + gen 
            observe(prob_equals(enc, c))
        end 

        text = [9, 11, 8, 21, 20, 21, 2, 5, 2, 12, 13, 2, 22, 9, 11, 8, 0, 11, 20, 6, 6, 2, 7, 0, 5, 20, 7, 0, 14, 20, 0, 24, 12, 9, 9, 5, 12, 20, 11, 24, 20, 7, 24, 17, 9, 11, 24, 12, 12, 2]
        for char in text
            sendChar(key, DistUIntOH{26}(char))
        end 
        key 
    end  

    debug_info_ref = Ref{CuddDebugInfo}()
    pr(c, algo=Cudd(debug_info_ref=debug_info_ref))
    println("NUM_NODES_START")
    println(debug_info_ref[].num_nodes)
    println("NUM_NODES_END")
end 

fun()

