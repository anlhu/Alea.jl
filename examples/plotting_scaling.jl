using Revise
using Dice
using Dates
using Plots

num_nodes_all = Int[] 
indices = Int[] 

for i in 1:18       # too long past 23 for ordering by global_id
    a = uniform(DistUInt{i + 1}, i)
    b = uniform(DistUInt{i + 1}, i)
    c = a + b

    start_time = now()
    nodes = num_nodes(c)
    elapsed = now() - start_time

    if Dates.value(elapsed) > 120 * 10^9  # 120 seconds in nanoseconds
        println("num_nodes took too long (>2 minutes). Exiting loop.")
        break
    else
        push!(indices, i)
        push!(num_nodes_all, nodes)
        println("\tNUM NODES ADD $(i + 1), $i = $nodes")
    end
end

println("Num Nodes: ", num_nodes_all)

plot(
    indices, 
    num_nodes_all,
    xlabel = "i (Input Size)",
    ylabel = "num_nodes",
    title = "num_nodes vs Input Size",
    marker = :circle,
    legend = false,
    grid = true
)