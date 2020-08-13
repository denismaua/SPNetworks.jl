# Runs MAP Inference algorithms
using SumProductNetworks
import SumProductNetworks: leaves, isleaf, issum, isprod, IndicatorFunction
import SumProductNetworks.MAP: maxproduct!, beliefpropagation! 
# import GraphicalModels: FactorGraph, FGNode, VariableNode, FactorNode
# import GraphicalModels.MessagePassing: HybridBeliefPropagation, update!, decode, marginal, setevidence!, setmapvar!, rmevidence!

# spn_filename = ARGS[1]
# spn_filename = "/Users/denis/code/SumProductNetworks/assets/example.pyspn.spn"
# spambase
# spn_filename = "/Users/denis/learned-spns/spambase/spambase.spn"
# query_filename = "/Users/denis/learned-spns/spambase/spambase.query"
# evid_filename = "/Users/denis/learned-spns/spambase/spambase.evid"
# nltcs
spn_filename = "/Users/denis/learned-spns/nltcs/nltcs.spn2"
query_filename = "/Users/denis/learned-spns/nltcs/nltcs.query"
evid_filename = "/Users/denis/learned-spns/nltcs/nltcs.evid"
# hepatitis
# spn_filename = "/Users/denis/learned-spns/hepatitis/hepatitis.spn2"
# query_filename = "/Users/denis/learned-spns/hepatitis/hepatitis.query"
# evid_filename = "/Users/denis/learned-spns/hepatitis/hepatitis.evid"
# ionosphere
# spn_filename = "/Users/denis/learned-spns/ionosphere/ionosphere.spn2"
# query_filename = "/Users/denis/learned-spns/ionosphere/ionosphere.query"
# evid_filename = "/Users/denis/learned-spns/ionosphere/ionosphere.evid"
# mushrooms
# spn_filename = "/Users/denis/learned-spns/mushrooms/mushrooms.spn2"
# query_filename = "/Users/denis/learned-spns/mushrooms/mushrooms.query"
# evid_filename = "/Users/denis/learned-spns/mushrooms/mushrooms.evid"
# dna
# spn_filename = "/Users/denis/learned-spns/dna/dna.spn2"
# query_filename = "/Users/denis/learned-spns/dna/dna.query"
# evid_filename = "/Users/denis/learned-spns/dna/dna.evid"
# nips
# spn_filename = "/Users/denis/learned-spns/nips/nips.spn2"
# query_filename = "/Users/denis/learned-spns/nips/nips.query"
# evid_filename = "/Users/denis/learned-spns/nips/nips.evid"

println("SPN: ", spn_filename)

# Load SPN from spn file (assume 0-based indexing)
loadtime = @elapsed spn = SumProductNetwork(spn_filename; offset = 1)
println("Loaded in $(loadtime)s ", summary(spn))
nvars = length(scope(spn))

maxch = 0
for node in spn
    if !isleaf(node)
        global maxch = max(maxch, length(node.children))
    end
end
# if maxch > 4
#     @warn "maximum node indegree too large: $maxch. It is highly recommend to split nodes before running this."
# end

# Load evidence and query variables
qfields = split(readline(query_filename))
nquery = parse(Int,qfields[1])
query = Set(map(f -> (parse(Int, f)+1), qfields[2:end]))
#println(query)
@assert length(query) == nquery

x = Array{Float64}(undef, nvars)
fill!(x, NaN) # marginalize non-query, non-evidence

efields = split(readline(evid_filename))
evidence = Dict{Int,Int}()
nevid = parse(Int,efields[1])
for i=2:2:length(efields)
    var = parse(Int,efields[i]) + 1
    value = parse(Int,efields[i+1]) + 1
    x[var] = value
    evidence[var] = value
end
@assert length(evidence) == nevid

#println(x)

# Run max-product 
mptime = @elapsed maxproduct!(x, spn, query)
mpvalue = spn(x)
printstyled("\nMaxProduct: "; color = :green)
println("$mpvalue [took $(mptime)s]\n")

# Run hybrid belief propagation
bptime = @elapsed beliefpropagation!(x, spn, query; maxiterations = 3, lowerbound = true, rndminit = false)
bpvalue = spn(x)
printstyled("\nBeliefPropagation: "; color = :green)
println("$bpvalue [took $(bptime)s]\n")

printstyled("Best Ratio: "; color = :green)
println( bpvalue/mpvalue )

# Translate SPN into distribution-equivalent Factor Graph
# println("Generating factor graph...")
# @time fg = spn2bn(spn)

# # consistency checks
# @assert length(fg.variables) == (length(spn) + nvars)
# @assert length(fg.factors) == length(spn)

# # Initialize belief propagation
# bp = HybridBeliefPropagation(fg; rndinit = false) # rndinit = true generates random initial message; = true sets all to 1.
# bp.normalize = true # normalize messages (sum = 1)?
# # Set evidence and query
# setevidence!(bp, "Y1", 2)
# for (v,e) in evidence
#     setevidence!(bp, "X$v", e)
# end
# for v in query
#     setmapvar!(bp, "X$v")
# end
# # run Inference
# println("Running belief propagation for $maxiterations iterations")
# cpad(s, n::Integer, p=" ") = rpad(lpad(s,div(n+length(s),2),p),n,p) # for printing centered
# columns = ["it", "time (s)", "residual", "value", "best?"]
# widths = [4, 10, 10, 24, 7]
# bordercolor = :light_cyan
# headercolor = :cyan
# fieldcolor = :normal
# printstyled('╭', join(map(w -> repeat('─',w), widths), '┬'), '╮', '\n' ;color = bordercolor)
# for (col,w) in zip(columns,widths)
#     printstyled("│"; color = bordercolor)
#     printstyled(cpad(col, w); color = headercolor)
# end
# printstyled("│\n"; color = bordercolor)
# printstyled('├', join(map(w -> repeat('─',w), widths), '┼'), '┤', '\n' ;color = bordercolor)
# # value of best incumbent solution
# # best = -Inf
# best = mpvalue
# start = time_ns()
# for it=1:maxiterations
#     # if it == 1
#     #     for v in query
#     #         setevidence!(bp, "X$v", x[v])
#     #     end
#     # elseif it == 2
#     #     for v in query
#     #         rmevidence!(bp, "X$v")
#     #     end
#     # end
#     # downward propagation  
#     res = 0.0
#     for i = length(spn):-1:1
#         node = spn[i]
#         # compute incoming messages from children
#         if isleaf(node) 
#             factor = fg.factors[string(i)]           
#             update!(bp, fg.variables["X$(node.scope)"], factor)
#             res = max(res, update!(bp, factor, fg.variables["Y$i"]))
#         else
#             factor = fg.factors[string(i)]
#             for j in node.children
#                 update!(bp, fg.variables["Y$j"], factor)
#             end
#             res = max(res, update!(bp, factor, fg.variables["Y$i"]))
#         end
#     end
#     # upward propagation
#     for i = 1:length(spn)
#         node = spn[i]
#         # compute outgoing messages to children
#         if isleaf(node)
#             factor = fg.factors[string(i)]
#             update!(bp, fg.variables["Y$i"], factor)
#             res = max(res, update!(bp, factor, fg.variables["X$(node.scope)"]))
#         else
#             factor = fg.factors[string(i)]
#             update!(bp, fg.variables["Y$i"], factor)
#             for j in node.children
#                 res = max(res, update!(bp, factor, fg.variables["Y$j"]))
#             end
#         end
#     end
#     # res = update!(bp)  
#     etime = (time_ns()-start)/1e9;
#     # prob = marginal(bp,"Y1")[2]
#     for i in query
#         x[i] = decode(bp, "X$i")
#     end
#     prob = spn(x)
#     for (col,w) in zip([it, round(etime, digits=2), round(res,digits=2), prob, prob >= best ? "*" : " "],widths)
#         printstyled("│"; color = bordercolor)
#         printstyled(lpad(col, w-1), ' '; color = fieldcolor)
#     end 
#     printstyled("│\n"; color = bordercolor)
#     global best = max(best, prob)
#     if res < 0.001 break end # early stop
# end
# printstyled('╰', join(map(w -> repeat('─',w), widths), '┴'), '╯', '\n' ;color = bordercolor)

# printstyled("Best Ratio: "; color = :green)
# println( best/mpvalue )
