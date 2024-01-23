using Plots
using Gen
using Distributions 
 
function init_func_state(n::Int, xs::Vector{Float64}, ys::Vector{Float64}, init_speed, init_angle, move_from_true)
    return Dict( "true_xs" => xs, "true_ys" => ys, "x" => xs[1], "y" => ys[1], "num_to_generate" => n, "angle" => init_angle, "speed" => init_speed, "output_x" => Vector{Float64}(), "output_y" => Vector{Float64}(), "num_dots_generated" => 0, "c"=>1, "continue_count"=>0, "move_from_true"=>move_from_true)
end

function normalize(dist::Vector{Float64})
    return dist/sum(dist)
end

function sample_categorical(probs::Vector{Float64})
    u = rand()
    cdf = cumsum(probs)
    for (i, c) in enumerate(cdf)
        if u < c return i end
    end
end

#turn a dict of number - probability pairs into a normalized distribution (arg to categorical )
function dict_to_dist(prob_dict::Dict, all_nodes::Vector{String})
    dist = Vector{Float64}()
    nodes_in_dict = collect(keys(prob_dict))
    for node_type in all_nodes
        if node_type in nodes_in_dict
            p = prob_dict[node_type]
        else
            p = 0
        end
        append!(dist, p)
    end
    dist = normalize(dist)
    return dist
end


function render_trace(trace; show_data=true)
    # Pull out xs from the trace
    xs, = get_args(trace)
    
    xmin = minimum(xs)
    xmax = maximum(xs)

    # Pull out the return value, useful for plotting
    func = get_retval(trace)

    fig = plot()

    if show_data
        xs = [trace[(:x, i)] for i=1:length(xs)]
        xs_model = evaluate_function(func, xs[1], 1., length(xs))
        println(func)
        println(xs)
        println(xs_model)
        # Plot the data set
        scatter!(1:length(xs), xs_model, c="black", label=nothing)
    end
    
    return fig
end;

function grid(renderer::Function, traces)
    Plots.plot(map(renderer, traces)...)
end;

function round_all(xs::Vector{Float64}; n=2)
    map(x -> round(x; digits=n), xs)
end

function perm_visualize(pred_xs::Vector{Float64}, pred_ys::Vector{Float64}, fig, c="red")
    gui(scatter!(fig, pred_xs, pred_ys, c=c, label=nothing))
end

function visualize_init(xs,ys)
    Plots.CURRENT_PLOT.nullableplot = nothing
    xmin=minimum(xs)
    xmax=maximum(xs)
    x_diff = xmax-xmin+1
    ymin=minimum(ys)
    ymax=maximum(ys)
    y_diff = ymax-ymin+1
    fig = plot!(xs, ys, color="black", xlim=(xmin-x_diff,xmax+x_diff), ylim=(ymin-y_diff,ymax+y_diff))
    gui(fig)
    gui(scatter!(fig, xs, ys, c="black", label=nothing))
    return fig
end


function node2str(node)
    str = string(node.tp) * "("
    n_children = length(node.children)
    for i=1:n_children
        str = str * node2str(node.children[i])
        if i != n_children
            str = str * ", "
        end
    end
    n_params = length(node.params)
    for i=1:n_params
        str = str * string(node.params[i])
        if i != n_params
            str = str * ", "
        end
    end
    str = str * ")"
    return str
end


function str2node(str, parent_tp="", parent_c=-1)
    idx =  findfirst("(", str)[1]
    tp = str[1:idx-1]
    rest = str[idx+1:end-1]
    child_list = Vector{Node}()
    if rest == ""
        #no children or params
        return Node(tp, parent_tp, parent_c, [], child_list)
    elseif !(occursin("(",rest))
        #no children, only params (rn we only allow for 1)
        if tp=="real"
            return Node(tp, parent_tp, parent_c, [parse(Float64, rest)], child_list)
        elseif tp=="int"
            return Node(tp, parent_tp, parent_c, [parse(Int64, rest)], child_list)
        else
            @assert false
        end
    else
        #children
        child_ct = 1
        while length(rest) > 0
            node_end = get_node_end(rest)
            child_str = rest[1:node_end]
            push!(child_list, str2node(child_str, tp, child_ct))
            child_ct = child_ct + 1
            rest = rest[node_end+3:end]
        end
        return Node(tp, parent_tp, parent_c, [], child_list)
    end
end

function get_node_end(str)
    #get idx bounds of first node in str
    open_ct = 0
    close_ct = 0
    for (i,ch) in enumerate(str)
        ch = string(ch)
        if cmp(ch, "(")==0
            open_ct += 1
        end
        if cmp(ch, ")")==0
            close_ct += 1
        end
        if open_ct > 0 && open_ct==close_ct
            return i
        end
    end
end
