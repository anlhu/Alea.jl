using Test
using Alea
using Alea: DistUInt

@testset "Bit‐Utility routines" begin

    # popcount(x): should count all 1 bits
    @testset "popcount" begin
        x = DistUInt{8}(0x00)         # 0000_0000 → 0 ones
        @test popcount(x) == 0

        x = DistUInt{8}(0xFF)         # 1111_1111 → 8 ones
        @test popcount(x) == 8

        x = DistUInt{8}(0xAA)         # 1010_1010 → 4 ones
        @test popcount(x) == 4

        x = DistUInt{8}(0x0F)         # 0000_1111 → 4 ones
        @test popcount(x) == 4
    end

    # trailing_zeros(x): count zeros starting from LSB until first one
    @testset "trailing_zeros" begin
        x = DistUInt{4}([false, false, true, true])  # bits 0b0011 → tz = 0
        @test trailing_zeros(x) == 0

        x = DistUInt{4}([true, false, true, false])  # bits 0b1010 → tz = 1
        @test trailing_zeros(x) == 1

        x = DistUInt{4}([false, true, false, false]) # bits 0b0100 → tz = 2
        @test trailing_zeros(x) == 2

        x = DistUInt{4}([false, false, false, false])# bits 0b0000 → tz = width = 4
        @test trailing_zeros(x) == 4
    end

    # logical right‐shift: insert zeros on the left, drop rightmost bits
    @testset ">>(right shift)" begin
        x = DistUInt{6}([1,0,1,1,0,1] .== 1)  # 0b101101 → 45
        y = x >> 0
        @test y.bits == x.bits               # shifting by 0 is identity

        y = x >> 1      
        exp_val =  DistUInt{6}(0b010110)                    # 0b010110 → 22
        @test y.bits == exp_val.bits

        y = x >> 3   
        exp_val = DistUInt{6}(0b000101)                     # 0b000101 → 5
        @test y.bits == exp_val.bits

        y = x >> 6                           # shift by width → all zeros
        exp_val = zero(DistUInt{6})
        @test y.bits == exp_val.bits
    end

    # division by power of two should use >> internally
    @testset "/ (power‐of‐two division)" begin
        # 15 ÷ 1 = 15, shift by 0
        a = DistUInt{5}(15)     # 0 1 1 1 1
        b = DistUInt{5}(1)
        exp_val = DistUInt{5}(15)
        @test (a / b).bits == exp_val.bits

        # 15 ÷ 2 = 7  (0b01111) 
        b = DistUInt{5}(2)
        exp_val = DistUInt{5}(7)
        @test (a / b).bits == exp_val.bits

        # 15 ÷ 4 = 3  (0b00011)
        b = DistUInt{5}(4)
        exp_val = DistUInt{5}(3)
        @test (a / b).bits == exp_val.bits

        # 15 ÷ 8 = 1
        b = DistUInt{5}(8)
        exp_val = DistUInt{5}(1)
        @test (a / b).bits == exp_val.bits

        # boundary: shifting by full width gives zero
        a = DistUInt{5}(7)
        b = DistUInt{5}(16)  # 2^4
        exp_val = zero(DistUInt{5})
        @test (a / b).bits == exp_val.bits
    end

end