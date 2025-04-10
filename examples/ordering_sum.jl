using Revise
using Dice

d = uniform(DistUInt{3}, 2)
e = uniform(DistUInt{3}, 2)
f = d+e
input_names = string.(get_flips(f) .|> x -> x.global_id)
print("\tNUM NODES ADD {3}, 2 = ", num_nodes(f))                # Should be 13
# dump_dot(f, inames=input_names, filename="threexthree.dot")

a = flip(0.5)
b = DistUInt( [flip(0.0), a, flip(0.5), flip(0.5), flip(0.5), flip(0.7)] )
c = DistUInt( [flip(0.0), flip(0.2), flip(0.5), flip(0.5), flip(0.5), a] )
sum = c+b
input_names = string.(get_flips(sum) .|> x -> x.global_id)
dump_dot(sum, inames=input_names, filename="multiple_indices.dot")
println("\tNum Nodes interdependent: ", num_nodes(sum))


a = uniform(DistUInt{4}, 3)
b = uniform(DistUInt{4}, 3)
c = a+b
input_names = string.(get_flips(c) .|> x -> x.global_id)
print("\tNUM NODES ADD {4}, 3 = ", num_nodes(c))
