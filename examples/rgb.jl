using Revise
using Dice

R = uniform(DistUInt{8}, 7)
G = uniform(DistUInt{8}, 7)
B = uniform(DistUInt{8}, 7)
four = DistUInt{8}(4)  # Binary 00000100
two  = DistUInt{8}(2)  # Binary 00000010
one = DistUInt{8}(1)

Z = R + G + B
# input_names_colors = string.(get_flips(Z) .|> x -> x.global_id)
# dump_dot(Z, inames=input_names_colors,filename="no-division-coloring.dot")
W = (R / four) + (G / two) + (B / four)
# input_names_colors_2 = string.(get_flips(W) .|> x -> x.global_id)
# dump_dot(W, inames=input_names_colors_2, filename="just-division-coloring.dot")
print("\tZ = NUM NODES: ", num_nodes(Z))
print("\tW = NUM NODES = ", num_nodes(W))