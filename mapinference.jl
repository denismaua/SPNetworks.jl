# Runs MAP Inference algorithms
using SumProductNetworks
import SumProductNetworks: leaves, isleaf, issum, isprod, IndicatorFunction
import SumProductNetworks.MAP: maxproduct!, lsmax!, beliefpropagation! 
# import GraphicalModels: FactorGraph, FGNode, VariableNode, FactorNode
# import GraphicalModels.MessagePassing: HybridBeliefPropagation, update!, decode, marginal, setevidence!, setmapvar!, rmevidence!

# Which algorithms to run?
# - mp: max-product
# - ls: Local-Search
# - bp: belief-propagation
# each algorithm uses the incumbent solution of previously ran algorithms (so order matters)
algorithms = [:mp, :bp]

# spn_filename = ARGS[1]
# spn_filename = "/Users/denis/code/SPN/mushrooms.spn2"
# in_filename = "/Users/denis/code/SPN/mushrooms_scenarios.map"
# spn_filename = "/Users/denis/code/SPN/nips.spn2"
# spn_filename = "/Users/denis/code/SPN/bag_50_nips.spn2"
spn_filename = "/Users/denis/learned-spns/nips/nips.spn2"
in_filename = "/Users/denis/code/SPN/nips.map"
# in_filename = "/Users/denis/code/SPN/nips_scenarios.map"
# nltcs
# spn_filename = "/Users/denis/learned-spns/nltcs/nltcs.spn2"
# query_filename = "/Users/denis/learned-spns/nltcs/nltcs.query"
# evid_filename = "/Users/denis/learned-spns/nltcs/nltcs.evid"
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
# assignment vector
x = Array{Float64}(undef, nvars)

# maxch = 0
# for node in spn
#     if !isleaf(node)
#         global maxch = max(maxch, length(node.children))
#     end
# end
# if maxch > 4
#     @warn "maximum node indegree too large: $maxch. It is highly recommend to split nodes before running this."
# end

# Load query, evidence and marginalized variables
open(in_filename) do io
    inst = 1
    while !eof(io)
        printstyled(inst; color = :red)
        # Read query variables
        fields = split(readline(io))
        header = fields[1]
        @assert header == "q"
        query = Set(map(f -> (parse(Int, f)+1), fields[2:end]))
        # nquery = parse(Int,qfields[1])
        # query = Set(map(f -> (parse(Int, f)+1), qfields[2:end]))
        #println(query)
        # @assert length(query) == nquery
        # fill!(x, NaN) # marginalize non-query, non-evidence
        # Read marginalized variables
        fields = split(readline(io))
        header = fields[1]
        @assert header == "m"
        marg = Set(map(f -> (parse(Int, f)+1), fields[2:end]))
        for i in marg
            x[i] = NaN
        end
        # Read evidence
        fields = split(readline(io))
        evidence = Dict{Int,Int}()
        header = fields[1]
        @assert header == "e"
        # nevid = parse(Int,efields[1])
        for i=2:2:length(fields)
            var = parse(Int,fields[i]) + 1
            value = parse(Int,fields[i+1]) + 1
            x[var] = value
            evidence[var] = value
        end
        # @assert length(evidence) == nevid
        # println(
        #     join(
        #         setdiff(
        #             collect(1:nvars), 
        #             union(query,keys(evidence))
        #             ) .- 1,
        #         " "
        #         )
        # )
        @assert (length(query) + length(marg) + length(evidence)) == nvars
        #println(x)

        for algo in algorithms
            if algo == :mp
                # Run max-product 
                mptime = @elapsed maxproduct!(x, spn, query)
                mpvalue = spn(x)
                printstyled("\nMaxProduct: "; color = :green)
                print("$mpvalue")
                printstyled("  [took $(mptime)s]\n"; color = :light_black)
            elseif algo == :ls
                # Run local search
                lstime = @elapsed lsmax!(x, spn, query)
                lsvalue = spn(x)
                printstyled("\nLocalSearch: "; color = :green)
                print("$lsvalue")
                printstyled("  [took $(lstime)s]\n"; color = :light_black)
            elseif algo == :bp
                # Run hybrid belief propagation
                bptime = @elapsed beliefpropagation!(x, spn, query; maxiterations = 5, lowerbound = true, rndminit = false)
                bpvalue = spn(x)
                printstyled("\nBeliefPropagation: "; color = :green)
                print("$bpvalue")
                printstyled("  [took $(bptime)s]\n"; color = :light_black)
                printstyled("Ratio: "; color = :green)
                ratio = bpvalue/mpvalue
                println( ratio )
            end
        end
        # for i = 1:10
        #     bptime = @elapsed beliefpropagation!(x, spn, query; maxiterations = 10, lowerbound = true, rndminit = true)
        #     bpvalue = spn(x)
        #     printstyled("\nBeliefPropagation: "; color = :green)
        #     print("$bpvalue")
        #     printstyled("  [took $(bptime)s]\n"; color = :light_black)
        #     ratio = bpvalue/mpvalue
        #     println( ratio )
        # end    

        # for debuggins, remove this
        inst += 1
        if inst > 3
            break
        end
    end
end

