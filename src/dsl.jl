using MacroTools: prewalk, postwalk
using IRTools
using IRTools: @dynamo, IR, recurse!, self, xcall, functional

export @alea_ite, @alea, alea, observe, constraint, assert_alea, @code_ir_alea

##################################
# Control flow macro
##################################

"Syntactic macro to make if-then-else supported by alea"
macro alea_ite(code)
    postwalk(esc(code)) do x
        if x isa Expr && (x.head == :if || x.head == :elseif)
            @assert length(x.args) == 3 "@alea_ite macro only supports purely functional if-then-else"
            ite_guard = gensym(:ite)
            return :(begin $ite_guard = $(x.args[1])
                    if ($(ite_guard) isa Dist{Bool})
                        ifelse($(ite_guard), $(x.args[2:3]...))
                    else
                        (if $(ite_guard)
                            $(x.args[2])
                        else
                            $(x.args[3])
                        end)
                    end
                end)
        end
        return x
    end
end

##################################
# Control flow + error + observation dynamo
##################################

"Interpret alea code with control flow, observations, and errors"
function alea(f) 
    dyna = AleaDyna()
    x = dyna(f)
    MetaDist(x, dyna.errors, dyna.observations)
end

"Interpret alea code with control flow, observations, and errors"
macro alea(code)
    esc(quote
        alea() do
            $code
        end
    end)
end


struct AleaDyna
    path::Vector{AnyBool}
    errors::Vector{Tuple{AnyBool, ErrorException}}
    observations::Vector{AnyBool}
    AleaDyna() = new(AnyBool[], Tuple{AnyBool, String}[], AnyBool[])
end

"Assert that the current code must be run within an @alea evaluation"
assert_alea() = error("This code must be called from within an @alea evaluation.")

observe(_) = assert_alea()

global dynamoed = Vector()

@dynamo function (dyna::AleaDyna)(a...)
    ir, time, _ = @timed begin
        ir = IR(a...)
        (ir === nothing) && return
        ir = functional(ir)
        ir = prewalk(ir) do x
            if x isa Expr && x.head == :call
                return xcall(self, x.args...)
            end
            return x
        end
        ir
    end
    global dynamoed
    push!(dynamoed, (time, a[1]))
    return ir
end

reset_dynamoed() = begin
    global dynamoed
    empty!(dynamoed)
end

top_dynamoed() = sort(dynamoed; by = x -> x[1], rev = true)

(::AleaDyna)(::typeof(assert_alea)) = nothing

(::AleaDyna)(::typeof(IRTools.cond), guard, then, elze) = IRTools.cond(guard, then, elze)

function (dyna::AleaDyna)(::typeof(IRTools.cond), guard::Dist{Bool}, then, elze)
    push!(dyna.path, guard)
    t = then()
    pop!(dyna.path)
    push!(dyna.path, !guard)
    e = elze()
    pop!(dyna.path)
    ifelse(guard, t, e)
end

path_condition(dyna) = reduce(&, dyna.path; init=true)

# TODO catch Base exceptions in ifelse instead
(dyna::AleaDyna)(::typeof(error), msg) =
    push!(dyna.errors, (path_condition(dyna), ErrorException(msg)))

(dyna::AleaDyna)(::typeof(observe), x) =
    push!(dyna.observations, !path_condition(dyna) | x)

(::AleaDyna)(::typeof(==), x::Dist, y) = 
    prob_equals(x,y)

(::AleaDyna)(::typeof(==), x, y::Dist) = 
    prob_equals(x,y)

(::AleaDyna)(::typeof(==), x::Dist, y::Dist) = 
    prob_equals(x,y)

# avoid transformation when it is known to trigger a bug
for f in :[getfield, typeof, Core.apply_type, typeassert, (===),
        Core.sizeof, Core.arrayset, tuple, isdefined, fieldtype, nfields,
        isa, Core.arraysize, repr, print, println, Base.vect, Broadcast.broadcasted,
        Broadcast.materialize, Core.Compiler.return_type, Base.union!, Base.getindex, Base.haskey,
        Base.pop!, Base.setdiff, unsafe_copyto!, Base.fma_emulated].args
    @eval (::AleaDyna)(::typeof($f), args...) = $f(args...)
end

# avoid transformation for performance (may cause probabilistic errors to become deterministic)
for f in :[xor, atleast_two, prob_equals, (&), (|), (!), isless, ifelse, 
    Base.collect_to!, Base.collect, Base.steprange_last, oneunit, 
    Base.pairwise_blocksize, eltype, firstindex, iterate, 
    continuous, uniform, flip].args
    @eval (::AleaDyna)(::typeof($f), args...) = $f(args...)
end

"Show pseudo-IR as run by alea's dynamo"
macro code_ir_alea(code)
    esc(quote
        ir = @code_ir $code
        functional(ir)
    end)
end