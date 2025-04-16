export pr, pr_marginals, condprobs, condprob, Cudd, CuddDebugInfo, ProbException, allobservations, JointQuery, returnvalue, expectation, variance

using DataStructures: DefaultDict

##################################
# Core inference API implemented by backends
##################################

"A choice of probabilistic inference algorithm"
abstract type InferAlgo end

"An error with a probabilistic condition"
const CondError = Tuple{AnyBool, ErrorException}

struct JointQuery
    # use a vector so that order is maintained, might be important for some inference heuristics
    bits::Vector{Dist{Bool}}
    JointQuery(bits) = new(unique!(bits))
end

"""
Compute probabilities of queries, optionally given evidence, 
conditional errors, and a custom inference algorithm.
"""
function pr(queries::Vector{JointQuery}; evidence::AnyBool = true, 
            errors::Vector{CondError} = CondError[], 
            algo::InferAlgo = default_infer_algo()) 
    pr(algo, evidence, queries; errors)
end

function pr(queries::JointQuery...; kwargs...)
    ans = pr(collect(queries); kwargs...)
    length(queries) == 1 ? ans[1] : ans
end

function pr(queries...; kwargs...)
    joint_queries = map(queries) do query
        JointQuery(tobits(query))
    end
    queryworlds = pr(collect(joint_queries); kwargs...)
    ans = map(queries, queryworlds) do query, worlds
        dist = DefaultDict(0.0)
        for (world, p) in worlds
            dist[frombits(query, world)] += p
        end
        dist
    end
    length(queries) == 1 ? ans[1] : ans
end

##################################
# Inference with metadata distributions from DSL
##################################
       
"A distribution computed by a alea program with metadata on observes and errors"
struct MetaDist
    returnvalue
    errors::Vector{CondError}
    observations::Vector{AnyBool}
end

returnvalue(x) = x.returnvalue
observations(x) = x.observations
allobservations(x) = reduce(&, observations(x); init=true)

function pr(x::MetaDist; ignore_errors=false, 
            algo::InferAlgo = default_infer_algo())
    evidence = allobservations(x)
    errors = ignore_errors ? CondError[] : x.errors
    pr(returnvalue(x); evidence, errors, algo)
end

function pr_marginals(x::MetaDist; ignore_errors=false, 
    algo::InferAlgo = default_infer_algo())
    evidence = allobservations(x)
    errors = ignore_errors ? CondError[] : x.errors
    pr(returnvalue(x)...; evidence, errors, algo)
end

function expectation(x::MetaDist; ignore_errors=false, 
    algo::InferAlgo = default_infer_algo())
evidence = allobservations(x)
errors = ignore_errors ? CondError[] : x.errors
expectation(returnvalue(x); evidence, errors, algo)
end

function variance(x::MetaDist; ignore_errors=false, 
    algo::InferAlgo = default_infer_algo())
evidence = allobservations(x)
errors = ignore_errors ? CondError[] : x.errors
variance(returnvalue(x); evidence, errors, algo)
end

##################################
# Prbabilistic exception support
##################################

"An error that is thrown with some non-zero probability"
const ProbError = Tuple{Float64, ErrorException}

"A collection of errors that is thrown with some non-zero probability"
struct ProbException <: Exception
    errors::Vector{ProbError}
end

function Base.showerror(io::IO, exc::ProbException)
    print(io, "ProbException: ")
    for (p, err) in exc.errors
        print(io, "  (")
        print(io, err)
        print(io, " with probability $p)")        
    end
end

##################################
# Inference backends
##################################

include("cudd.jl")