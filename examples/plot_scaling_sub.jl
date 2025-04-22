using Revise
using Dice
using Dates
using Plots

num_nodes_all = Int[] 
indices = Int[] 

for i in 1:10       # too long past 23 for ordering by global_id
    a = uniform(DistUInt{i + 1}, i)
    b = uniform(DistUInt{i + 1}, i)
    c = a - b

    nodes = num_nodes(c)

    push!(indices, i)
    push!(num_nodes_all, nodes)
    println("\tNUM NODES ADD $(i + 1), $i = $nodes")
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