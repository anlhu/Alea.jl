using Pkg; Pkg.activate(@__DIR__)
using Alea, Distributions

precision = 5
DFiP = DistFixedPoint{9+precision, precision}
num_pieces = 128

code = @alea begin

  isWeekend = flip(2/7)
  hour = if isWeekend
            continuous(DFiP, Normal(5, 4), num_pieces, 0.0, 8.0)
        else
            continuous(DFiP, Normal(2, 4), num_pieces, 0.0, 8.0)
        end
  observe(hour == DFiP(6.0))
  isWeekend
end

# HMC-estimated ground truth: 1.363409828
@time pr(code)