# Runs MAP Inference algorithms
using SumProductNetworks
import SumProductNetworks: leaves, isleaf, issum, isprod, IndicatorFunction
import SumProductNetworks.MAP: maxproduct!, localsearch!, beliefpropagation! 
# import GraphicalModels: FactorGraph, FGNode, VariableNode, FactorNode
# import GraphicalModels.MessagePassing: HybridBeliefPropagation, update!, decode, marginal, setevidence!, setmapvar!, rmevidence!

maxinstances = 100

# Which algorithms to run?
# - mp: max-product
# - ls: Local-Search
# - bp: belief-propagation
# each algorithm uses the incumbent solution of previously ran algorithms (so order matters)
# algorithms = [:mp, :ls, :bp]
# algorithms = [:mp, :bp, :ls]
algorithms = [:mp, :ls]
# algorithms = []

# collect results (value and runtime)
results = Dict{Symbol,Array{Float64}}()
for algo in algorithms
    results[algo] = Float64[]
end

# spn_filename = ARGS[1]

# mushrooms
# spn_filename = "/Users/denis/learned-spns/mushrooms/mushrooms.spn2"
# in_filename = "/Users/denis/code/SPN/mushrooms.map"
# spn_filename = "/Users/denis/code/SPN/mushrooms.spn2"
# spn_filename = "/Users/denis/code/SPN/mushrooms.spn"
spn_filename = "/Users/denis/code/SPN/bag_50_mushrooms.spn2"
in_filename = "/Users/denis/code/SPN/mushrooms_scenarios.map"

# dna
# spn_filename = "/Users/denis/learned-spns/dna/dna.spn2"
# in_filename = "/Users/denis/code/SPN/dna.map"

# nips
# spn_filename = "/Users/denis/code/SPN/nips.spn2"
# spn_filename = "/Users/denis/code/SPN/nips.spn"
# spn_filename = "/Users/denis/code/SPN/bag_50_nips.spn2"
# spn_filename = "/Users/denis/learned-spns/nips/nips.spn2"
# in_filename = "/Users/denis/code/SPN/nips.map"
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
# spn_filename = "/Users/denis/learned-spns/dna/dna.spn2"
# query_filename = "/Users/denis/learned-spns/dna/dna.query"
# evid_filename = "/Users/denis/learned-spns/dna/dna.evid"
# nips
# spn_filename = "/Users/denis/learned-spns/nips/nips.spn2"
# query_filename = "/Users/denis/learned-spns/nips/nips.query"
# evid_filename = "/Users/denis/learned-spns/nips/nips.evid"

println("SPN: ", spn_filename)
println("Query: ", in_filename)
println()

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
totaltime = @elapsed open(in_filename) do io
    inst = 1
    printstyled("╮\n"; color = :red)
    while !eof(io)
        # fill!(x, NaN) # reset configuration
        printstyled("├(", inst, ")\n"; color = :red)
        # Read query variables
        fields = split(readline(io))
        header = fields[1]
        @assert header == "q"
        query = Set(map(f -> (parse(Int, f)+1), fields[2:end]))
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
        mpvalue = 0.0
        for algo in algorithms
            printstyled("│\t"; color = :red)
            if algo == :mp
                # Run max-product 
                runtime = @elapsed maxproduct!(x, spn, query)
                printstyled("MaxProduct: "; color = :green)
            elseif algo == :ls
                # Run local search
                runtime = @elapsed localsearch!(x, spn, query, 100)
                printstyled("LocalSearch: "; color = :green)
            elseif algo == :bp
                # Run hybrid belief propagation
                runtime = @elapsed beliefpropagation!(x, spn, query; 
                    maxiterations = 5, 
                    lowerbound = false, 
                    warmstart = false,
                    rndminit = false)
                printstyled("BeliefPropagation: "; color = :green)
                # for i = 1:10
                #     bptime = @elapsed beliefpropagation!(x, spn, query; maxiterations = 10, lowerbound = true, rndminit = true)
                #     bpvalue = spn(x)
                #     printstyled("\nBeliefPropagation: "; color = :green)
                #     print("$bpvalue")
                #     printstyled("  [took $(bptime)s]\n"; color = :light_black)
                #     ratio = bpvalue/mpvalue
                #     println( ratio )
                # end    
            end
            value = spn(x)
            print("$value")
            printstyled("  [$(runtime)s]\n"; color = :light_black)
            # push!(results[algo], (value,runtime))
            push!(results[algo], value)
        end
        inst += 1
        # if maximum no. of instances is reached, stop
        if inst > maxinstances
            break
        end
    end
    printstyled("╯\n"; color = :red)
end
insts = length(results[algorithms[1]])
println("Total time: $(totaltime)s\nTotal instances: $insts")
cpad(s, n::Integer, p=" ") = rpad(lpad(s,div(n+length(s),2),p),n,p) # for printing centered
columns = [:it]
widths = [4]
for algo in algorithms
    push!(columns,algo)
    push!(widths,25)
end
bordercolor = :light_cyan
headercolor = :cyan
fieldcolor = :normal
printstyled('╭', join(map(w -> repeat('─',w), widths), '┬'), '╮', '\n' ; color = bordercolor)
for (col,w) in zip(columns,widths)
    printstyled("│"; color = bordercolor)
    printstyled(cpad(string(col), w); color = headercolor)
end
printstyled("│\n"; color = bordercolor)
printstyled('├', join(map(w -> repeat('─',w), widths), '┼'), '┤', '\n' ; color = bordercolor)
for it = 1:insts
    values = Any[ it ; [ results[algo][it] for algo in columns[2:end] ] ]
    for (col,width) in zip(values, widths)
        printstyled("│"; color = bordercolor)
        printstyled(lpad(col, width-1), ' '; color = fieldcolor)
    end 
    printstyled("│\n"; color = bordercolor)
end
printstyled('╰', join(map(w -> repeat('─',w), widths), '┴'), '╯', '\n' ; color = bordercolor)


