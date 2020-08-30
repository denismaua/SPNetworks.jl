# I/O functions

"""
Prints out Node content
"""
function Base.show(io::IO, n::SumNode)
    print(io,"+")
    for i=1:length(n.children)
        print(io, " $(n.children[i]) $(n.weights[i])")
    end
end

function Base.show(io::IO, n::ProductNode)
    print(io,"*")
    for i=1:length(n.children)
        print(io, " $(n.children[i])")
    end
end
function Base.show(io::IO, n::IndicatorFunction)
    print(io, "indicator $(n.scope) $(n.value)")
end
function Base.show(io::IO, n::CategoricalDistribution)
    print(io, "categorical $(n.scope)")
    for v in n.values
        print(io," $v")
    end
end
function Base.show(io::IO, n::GaussianDistribution)
    print(io, "indicator $(n.scope) $(n.mean) $(n.variance)")
end

""" 
    summary(io::IO, spn::SumProductNetwork)

Print out information about the network `spn` to stream `io`
"""
function Base.summary(io::IO, spn::SumProductNetwork)
    len, lensum, lenprod, lenleaves = length(spn), length(sumnodes(spn)), length(productnodes(spn)), length(leaves(spn))
    lenvars = length(scope(spn))
    print(io, "Sum-Product Network with ", len, (len==1 ? " node" : " nodes"), " (", lensum, (lensum==1 ? " sum" : " sums"), ", ", lenprod, (lenprod==1 ? " product" : " products"), ", ", lenleaves, (lenleaves==1 ? " leaf" : " leaves"), ") and ", lenvars, (lenvars==1 ? " variable" : " variables"))
    # print(io, "Sum-Product Network with $(length(spn)) nodes ($(length(sumnodes(spn))) sums, $(length(productnodes(spn))) products, $(length(leaves(spn))) leaves), $(nparams(spn)) parameters, and $(length(scope(spn))) variables")
    # #println(io, summary(spn))
    # println(io, "Sum-Product Network with:")
    # println(io, "│\t$(length(spn)) nodes: $(length(sumnodes(spn))) sums, $(length(productnodes(spn))) products, $(length(leaves(spn))) leaves")
    # println(io, "│\t$(nparams(spn)) parameters")
    # println(io, "╰\t$(length(scope(spn))) variables")
    #println(io, "\tdepth = $(length(layers(spn)))")
end
# function Base.summary(spn::SumProductNetwork) 
#     io = IOBuffer()
#     print(io, "Sum-Product Network with $(length(spn)) nodes ($(length(sumnodes(spn))) sums, $(length(productnodes(spn))) products, $(length(leaves(spn))) leaves), $(nparams(spn)) parameters, and $(length(scope(spn))) variables.")
#     String(take!(io))    
# end

""" 
    show(io::IO, spn::SumProductNetwork)

Print the nodes of the network `spn` to stream `io`
"""
function Base.show(io::IO, spn::SumProductNetwork) 
    println(io, "SumProductNetwork(IOBuffer(\"\"\"# ", summary(spn))
    for (i, node) in enumerate(spn)
        println(io, i, " ", node)
    end
    print(io,"\"\"\"))")
    # print(io, summary(spn))
end
function Base.show(io::IO, ::MIME"text/plain", spn::SumProductNetwork) 
    # recur_io = IOContext(io)
    recur_io = IOContext(io, :SHOWN_SET => spn)
    limit::Bool = get(io, :limit, false)
    if !haskey(io, :compact)
        recur_io = IOContext(recur_io, :compact => true)
    end
    summary(io, spn)
    print(io, ":")
    # print(io, "\n  1: ", spn[1])
    # println(io, "Sum-Product Network with:")
    # println(io, "│\t$(length(spn)) nodes: $(length(sumnodes(spn))) sums, $(length(productnodes(spn))) products, $(length(leaves(spn))) leaves")
    # println(io, "│\t$(nparams(spn)) parameters")
    # println(io, "╰\t$(length(scope(spn))) variables")
    if limit
        sz = displaysize(io)
        rows, cols = sz[1] - 3, sz[2]
        rows < 4   && (print(io, " …"); return)
        cols -= 5 # Subtract the width of prefix "  " and separator " : "
        cols < 12  && (cols = 12) # Minimum widths of 2 for id, 4 for value
        rows -= 1 # Subtract the summary

       # determine max id width to align the output, caching the strings
       ks = Vector{String}(undef, min(rows, length(spn)))
       vs = Vector{String}(undef, min(rows, length(spn)))
       keylen = 0
       vallen = 0
       for (i, n) in enumerate(spn)
           i > rows && break
           ks[i] = sprint(show, i, context=recur_io, sizehint=0)
           vs[i] = sprint(show, n, context=recur_io, sizehint=0)
           keylen = clamp(length(ks[i]), keylen, cols)
           vallen = clamp(length(vs[i]), vallen, cols)
       end
       if keylen > max(div(cols, 2), cols - vallen)
           keylen = max(cld(cols, 3), cols - vallen)
       end        
    else
        rows = cols = typemax(Int)
    end
    for (i, node) in enumerate(spn)
        print(io, "\n  ")
        if i == rows < length(spn)
            print(io, rpad("⋮", keylen), " : ⋮")
            # print(io, rpad("⋮", 2))
            break
        end
        if limit
            key = rpad(Base._truncate_at_width_or_chars(ks[i], keylen, "\r\n"), keylen)
        else
            key = sprint(show, i, context=recur_io, sizehint=0)
        end
        print(recur_io, key)
        print(io, " : ")
        if limit
            val = Base._truncate_at_width_or_chars(vs[i], cols - keylen, "\r\n")
            print(io, val)
        else
            show(recur_io, n)
        end
    end
end

"""
    SumProductNetwork(filename::AbstractString; offset=0)::SumProductNetwork
    SumProductNetwork(io::IO=stdin; offset=0)::SumProductNetwork

Reads network from file. Assume 1-based indexing for node ids and values at indicator nodes. Set offset = 1 if these values are 0-based instead.
"""
function SumProductNetwork(filename::String; offset::Integer = 0)
    spn = open(filename) do file
        spn = SumProductNetwork(file, offset=offset)
    end
    spn
end
function SumProductNetwork(io::IO=stdin; offset::Integer = 0)
    # create dictionary of node_id => node (so they can be read in any order)
    nodes = Dict{UInt,Node}()
    # read and create nodes
    for line in eachline(io)
        # remove line break
        line = strip(line)
        # remove comments
        i = findfirst(isequal('#'), line)
        if !isnothing(i)
            line = line[1:i-1]
        end
        if length(line) > 0 
            fields = split(line)
            if tryparse(Int, fields[1]) !== nothing
                nodeid = parse(Int,fields[1]) + offset
                nodetype = fields[2][1]
            else
                nodeid = parse(Int,fields[2]) + offset
                nodetype = fields[1][1]
            end
            if nodetype == '+'
                node = SumNode([ parse(Int,ch) + offset for ch in fields[3:2:end] ],
                               [ parse(Float64,w) for w in fields[4:2:end] ])
            elseif nodetype == '*'
                node = ProductNode([ parse(Int,id) + offset for id in fields[3:end] ])
            elseif nodetype == 'c'
                varid = parse(Int,fields[3]) + offset
                node = CategoricalDistribution(varid,[ parse(Float64,value) for value in fields[4:end] ])
            elseif nodetype == 'i' || nodetype == 'l'
                varid = parse(Int,fields[3]) + offset
                value = parse(Float64,fields[4]) + offset
                node = IndicatorFunction(varid,value)
            elseif nodetype == 'g'
                # TODO: read Gaussian leaves
                error("Reading of gaussian nodes is not implemented!")
            end
            nodes[nodeid] = node
        end
    end
    nodelist = Vector{Node}(undef, length(nodes))
    for (id,node) in nodes
        nodelist[id] = node
    end    
    spn = SumProductNetwork(nodelist)
    sort!(spn) # ensure nodes are topologically sorted (with ties broken by bfs-order)
    spn
end

"""
    save(spn,filename,[offset=0])

Writes network spn to file. Offset adds constant to node instances (useful for translating to 0 starting indexes).
"""
function save(spn::SumProductNetwork, filename::String, offset = 0)
    open(filename, "w") do io
        for i = 1:length(spn)
            println(io,"$i $(spn[i])")
        end
    end
end

"""
    todot(io, son)

Prints out network structure in graphviz format
"""
function todot(io::IO, spn::SumProductNetwork)
    println(io, "digraph S {")
    for i = 1:length(spn)
	      if isa(spn[i], SumNode)
            if i == 1
		            println(io, "n$i [shape=circle,rank=source,style=filled,color=\"#fed434\",label=\"+\",margin=0.05];")
            else
		            println(io, "n$i [shape=circle,style=filled,color=\"#fed434\",label=\"+\",margin=0.05];")
            end
	      elseif isa(spn[i], ProductNode)
		        println(io, "n$i [shape=circle,style=filled,color=\"#b0db51\",label=\"×\",margin=0.05];")
	      elseif isa(spn[i], LeafNode)
		        println(io, "n$i [shape=circle,rank=sink,style=filled,color=\"#02a1d8\",label=\"X$(spn[i].scope)\",margin=0.05];")
	      end
        if !isa(spn[i],LeafNode)
            print(io, "n$i -> { ")
            for j in children(spn,i)
                print(io, "n$j; ")
            end
            println(io, "};")
        end
    end
    println(io, "}")
end
