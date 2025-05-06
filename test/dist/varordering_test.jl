using Test
using Dice
using Dice: Flip, num_ir_nodes

@testset "VarOrder Interleaving - Addition" begin

    d = uniform(DistUInt{3}, 2)
    e = uniform(DistUInt{3}, 2)
    f = d+e
    @test num_nodes(f) == 13

    a = uniform(DistUInt{4}, 3)
    b = uniform(DistUInt{4}, 3)
    c = a+b
    @test num_nodes(c) == 22

    g = uniform(DistUInt{5}, 4)
    h = uniform(DistUInt{5}, 4)
    i = g+h
    @test num_nodes(i) == 31

    g = uniform(DistUInt{8}, 7)
    h = uniform(DistUInt{8}, 7)
    i = g+h
    @test num_nodes(i) == 58

    g = uniform(DistUInt{10}, 9)
    h = uniform(DistUInt{10}, 9)
    i = g+h
    @test num_nodes(i) == 76

    g = uniform(DistUInt{15}, 14)
    h = uniform(DistUInt{15}, 14)
    i = g+h
    @test num_nodes(i) == 121

    d = uniform(DistUInt{20}, 19)
    e = uniform(DistUInt{20}, 19)
    f = d+e
    @test num_nodes(f) == 166

    d = uniform(DistUInt{31}, 30)
    e = uniform(DistUInt{31}, 30)
    f = d+e
    @test num_nodes(f) == 265

end

@testset "VarOrder Interleaving - Subtraction" begin

    d = uniform(DistUInt{3}, 2)
    e = uniform(DistUInt{3}, 2)
    f = d-e
    @test num_nodes(f) == 15

    a = uniform(DistUInt{4}, 3)
    b = uniform(DistUInt{4}, 3)
    c = a-b
    @test num_nodes(c) == 27

    g = uniform(DistUInt{5}, 4)
    h = uniform(DistUInt{5}, 4)
    i = g-h
    @test num_nodes(i) == 42

    g = uniform(DistUInt{8}, 7)
    h = uniform(DistUInt{8}, 7)
    i = g-h
    @test num_nodes(i) == 105

    g = uniform(DistUInt{10}, 9)
    h = uniform(DistUInt{10}, 9)
    i = g-h
    @test num_nodes(i) == 162

    g = uniform(DistUInt{15}, 14)
    h = uniform(DistUInt{15}, 14)
    i = g-h
    @test num_nodes(i) == 357

    d = uniform(DistUInt{20}, 19)
    e = uniform(DistUInt{20}, 19)
    f = d-e
    @test num_nodes(f) == 627

end