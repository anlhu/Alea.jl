# The bridge between autodiff and cudd
export LogPr, compute_mixed, train!, pr_mixed, support_mixed, with_concrete_ad_flips

mutable struct LogPr <: ADNode
    bool::Dist{Bool}
end
NodeType(::Type{LogPr}) = Leaf()
compute_leaf(::LogPr) = error("LogPr must be expanded")
backward(::LogPr, _, _) = error("LogPr must be expanded")

mutable struct LogPrExpander
    w::WMC
    cache::Dict{ADNode, ADNode}
    function LogPrExpander(w)
        new(w, Dict{ADNode, ADNode}())
    end
end

function expand_logprs(l::LogPrExpander, root::ADNode)::ADNode
    fl(x::LogPr) = begin
        lpr = logprob(l.w, x.bool)
        if lpr isa AbstractFloat
            Constant(lpr)
        else
            expand_logprs(l, logprob(l.w, x.bool))
        end
    end
    fl(x::Var) = x
    fl(x::Constant) = x
    fi(x::Add, call) = Add(call(x.x), call(x.y))
    fi(x::Mul, call) = Mul(call(x.x), call(x.y))
    fi(x::Pow, call) = Pow(call(x.x), call(x.y))
    fi(x::Sin, call) = Sin(call(x.x))
    fi(x::Cos, call) = Cos(call(x.x))
    fi(x::Log, call) = Log(call(x.x))
    fi(x::ADMatrix, call) = ADMatrix(map(call, x.x))
    fi(x::GetIndex, call) = GetIndex(call(x.x), x.i)
    fi(x::Map, call) = Map(x.f, x.f′, call(x.x))
    fi(x::Transpose, call) = Transpose(call(x.x))
    fi(x::NodeLogPr, call) = NodeLogPr(call(x.pr), call(x.hi), call(x.lo))
    foldup(root, fl, fi, ADNode, l.cache)
end

# Within roots' LogPrs there are Dist{Bool} DAGs. Collect minimal roots all DAGs
function bool_roots(roots)
    # TODO: have non-root removal be done in src/inference/cudd/compile.jl
    seen_adnodes = Dict{ADNode, Nothing}()
    seen_bools = Dict{AnyBool, Nothing}()
    non_roots = Set{AnyBool}()
    to_visit = Vector{ADNode}(roots)
    while !isempty(to_visit)
        x = pop!(to_visit)
        foreach(x, seen_adnodes) do y
            if y isa LogPr
                foreach(y.bool, seen_bools) do bool
                    union!(non_roots, children(bool))
                    if bool isa Flip && bool.prob isa ADNode && !haskey(seen_adnodes, bool.prob)
                        push!(to_visit, bool.prob)
                    end
                end
            end
        end
    end
    setdiff(keys(seen_bools), non_roots)
end

function compute_mixed(var_vals::Valuation, root::ADNode)
    compute_mixed(var_vals, [root])[root]
end

function compute_mixed(var_vals::Valuation, roots)
    l = LogPrExpander(WMC(BDDCompiler(bool_roots(roots))))
    expanded_roots = [expand_logprs(l, x) for x in roots]
    vals = compute(var_vals, expanded_roots)
    Dict(root => vals[l.cache[root]] for root in roots)
end

function train!(
    var_vals::Valuation,
    loss_lrs::Vector{<:Pair{<:ADNode, <:Real}};
    epochs::Integer,
    append_last_loss=true,
    stop_if_inf_or_nan=true,
)
    # Unzip
    losses = ADNode[]
    lrs = Real[]
    for (loss, lr) in loss_lrs
        push!(losses, loss)
        push!(lrs, lr)
    end

    # Expand
    l = LogPrExpander(WMC(BDDCompiler(bool_roots(losses))))
    losses = [expand_logprs(l, loss) for loss in losses]

    curves = [[] for _ in 1:length(losses)]
    function update_curves(vals)
        for (i, loss) in enumerate(losses)
            push!(curves[i], vals[loss])
        end
    end

    for _ in 1:epochs
        vals, derivs = differentiate(
            var_vals,
            Derivs(loss => lr for (loss, lr) in zip(losses, lrs))
        )
        update_curves(vals)

        if stop_if_inf_or_nan && any(isinf(vals[loss]) || isnan(vals[loss]) for loss in losses)
            return curves
        end

        # update vars
        for (adnode, d) in derivs
            if adnode isa Var
                var_vals[adnode] -= d
            end
        end
    end

    append_last_loss && update_curves(compute(var_vals, losses))
    curves
end

function train!(
    var_vals::Valuation,
    loss::ADNode;
    epochs::Integer,
    learning_rate::Real,
    append_last_loss=true,
    stop_if_inf_or_nan=true,
)
    train!(var_vals, [loss => learning_rate]; epochs, append_last_loss, stop_if_inf_or_nan)[1]
end

function collect_flips(bools)
    flips = Vector{Flip}()
    foreach_down(bools) do x
        x isa Flip && push!(flips, x)
    end
    flips
end

function with_concrete_ad_flips(f, var_vals, dist)
    flip_to_original_prob = Dict()
    a = ADComputer(var_vals)
    l = LogPrExpander(WMC(BDDCompiler()))
    for x in collect_flips(tobits(dist))
        if x.prob isa ADNode
            flip_to_original_prob[x] = x.prob
            x.prob = compute(a, expand_logprs(l, x.prob))
        end
    end
    res = f()
    for (x, prob) in flip_to_original_prob
        x.prob = prob
    end
    res
end

function pr_mixed(var_vals)
    (args...; kwargs...) -> with_concrete_ad_flips(var_vals, args...) do
        pr(args...; kwargs...)
    end
end

function support_mixed(args...; kwargs...)
    flip_to_original_prob = Dict()
    for x in collect_flips(tobits(args...))
        if x.prob isa ADNode
            flip_to_original_prob[x] = x.prob
            x.prob = 0.5
        end
    end
    res = keys(pr(args...; kwargs...))
    for (x, prob) in flip_to_original_prob
        x.prob = prob
    end
    res
end