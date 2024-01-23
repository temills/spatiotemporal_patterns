import Random
using Distributions
using Gen
include("utils.jl")

""" This file contains the node types and distributions over types which are used in proposals
    e.g. the `expr_to_expr_c1_dist' gives a categorical distribution over expr nodes whose first child is an expr node.
    This is useful in add_node proposals,
    when we might want to insert a node between an expr node and its first child which is an expr node """

#abstract type Node end

struct Node
    tp::String
    parent_tp::String
    parent_c::Int
    params::Vector{Any}
    children::Vector{Node}
    size::Int
end
Base.size(node::Node) = node.size
Node(tp, parent_tp, parent_c, params, children::Vector{Node}) = Node(tp, parent_tp, parent_c, params, children, sum([c.size for c in children]) + 1)


function eval_node(node::Node, func_state::Dict)
    if node.tp in ["real", "int"]
        @assert length(node.params)==1
        return node.params[1]
    elseif node.tp == "var_t"
        return func_state["num_dots_generated"]
    elseif node.tp == "var_c"
        return func_state["c"]
    elseif node.tp == "var_x"
        return func_state["x"]
    elseif node.tp == "var_y"
        return func_state["y"]
    elseif node.tp == "var_angle"
        return func_state["angle"]
    elseif node.tp == "var_speed"
        return func_state["speed"]
    elseif node.tp == "round"
        return floor(eval_node(node.children[1], func_state))
    elseif node.tp == "sin"
        return sin(eval_node(node.children[1], func_state))
    elseif node.tp == "plus"
        return eval_node(node.children[1], func_state) + eval_node(node.children[2], func_state)
    elseif node.tp == "minus"
        return eval_node(node.children[1], func_state) - eval_node(node.children[2], func_state)
    elseif node.tp == "times"
        return eval_node(node.children[1], func_state) * eval_node(node.children[2], func_state)
    elseif node.tp == "divide"
        divisor = eval_node(node.children[2], func_state)
        if floor(divisor) == 0
            return 0
        else
            return eval_node(node.children[1], func_state) / divisor
        end
    elseif node.tp == "mod"
        divisor = eval_node(node.children[2], func_state)
        if floor(divisor) == 0
            return 0
        else
            return eval_node(node.children[1], func_state) % divisor
        end
    elseif node.tp == "change_x"
        dx = eval_node(node.children[1], func_state)
        func_state["x"] = func_state["x"] + dx
        return func_state
    elseif node.tp == "change_y"
        dy = eval_node(node.children[1], func_state)
        func_state["y"] = func_state["y"] + dy
        return func_state
    elseif node.tp == "set_x"
        x = eval_node(node.children[1], func_state)
        func_state["x"] = x
        return func_state
    elseif node.tp == "set_y"
        y = eval_node(node.children[1], func_state)
        func_state["y"] = y
        return func_state
    elseif node.tp == "change_angle"
        da = eval_node(node.children[1], func_state)
        func_state["angle"] = func_state["angle"] + da
        if func_state["angle"] >= 0
            func_state["angle"] = mod(func_state["angle"], 4)
        else
            func_state["angle"] = mod(func_state["angle"], -4)
        end
        return func_state
    elseif node.tp == "change_speed"
        ds = eval_node(node.children[1], func_state)
        func_state["speed"] = func_state["speed"] + ds
        return func_state
    elseif node.tp == "move"
        func_state["x"] = func_state["x"] + (func_state["speed"] * cos(func_state["angle"] * pi/2)) 
        func_state["y"] = func_state["y"] + (func_state["speed"] * sin(func_state["angle"]* pi/2))
        push!(func_state["output_x"], func_state["x"])
        push!(func_state["output_y"], func_state["y"])
        func_state["num_dots_generated"] = func_state["num_dots_generated"] + 1
        if func_state["move_from_true"] && func_state["num_dots_generated"] < length(func_state["true_xs"])
            func_state["x"] = func_state["true_xs"][func_state["num_dots_generated"]+1]
            func_state["y"] = func_state["true_ys"][func_state["num_dots_generated"]+1]
        end
        return func_state
    elseif node.tp == "dot"
        push!(func_state["output_x"], func_state["x"])
        push!(func_state["output_y"], func_state["y"])
        func_state["num_dots_generated"] = func_state["num_dots_generated"] + 1
        if func_state["move_from_true"] && func_state["num_dots_generated"] < length(func_state["true_xs"])
            func_state["x"] = func_state["true_xs"][func_state["num_dots_generated"]+1]
            func_state["y"] = func_state["true_ys"][func_state["num_dots_generated"]+1]
        end
        return func_state
    elseif node.tp == "repeat"
        n = min(eval_node(node.children[2], func_state), 10000)
        for i in 1:abs(n)
            if (i > func_state["num_to_generate"]) || (func_state["num_dots_generated"] > func_state["num_to_generate"])
                break
            end
            func_state = eval_node(node.children[1], func_state)
        end
        return func_state
    elseif node.tp == "concat"
        func_state = eval_node(node.children[1], func_state)
        func_state = eval_node(node.children[2], func_state)
        return func_state
    elseif node.tp == "subprogram"
        x, y, angle, speed = func_state["x"], func_state["y"], func_state["angle"], func_state["speed"]
        func_state = eval_node(node.children[1], func_state)
        func_state["x"], func_state["y"], func_state["angle"], func_state["speed"] = x, y, angle, speed
        return func_state
    elseif node.tp == "continue"
        #only allow for one "continue" per func
        if func_state["continue_count"] > 1
            return func_state
        end
        func_state["continue_count"] = func_state["continue_count"] + 1
        #only loop up to n_predictions time
        count = 0
        while (length(func_state["output_x"]) < func_state["num_to_generate"]) && (count < func_state["num_to_generate"])
            func_state = eval_node(node.children[1], func_state)
            count = count + 1
        end
        return func_state
    elseif node.tp == "increment_counter"
        func_state["c"] = func_state["c"] + 1
        return func_state
    else
        error("Unknown node type: $(node.tp)")
    end
end


function add_primitive(name::String, types, child_types)
    info[name] = Dict("types"=>types, "child_types"=>child_types)
    push!(node_list, name)
end
node_list = Vector{String}()
info = Dict()
add_primitive("real", ["expr"], [])
add_primitive("int", ["int_expr", "expr"], [])
add_primitive("var_t", ["int_expr", "expr"], [])
add_primitive("var_c", ["int_expr", "expr"], [])
add_primitive("var_x", ["expr"], [])
add_primitive("var_y", ["expr"], [])
add_primitive("var_speed", ["expr"], [])
add_primitive("var_angle", ["expr"], [])
add_primitive("round", ["int_expr", "expr"], ["expr"])
add_primitive("sin", ["expr"], ["expr"])
add_primitive("plus", ["expr"], ["expr", "expr"])
add_primitive("minus", ["expr"], ["expr", "expr"])
add_primitive("times", ["expr"], ["expr", "expr"])
add_primitive("divide", ["expr"], ["expr", "expr"])
add_primitive("mod", ["int_expr", "expr"], ["int_expr", "int_expr"])
add_primitive("change_x", ["op"], ["expr"])
add_primitive("change_y", ["op"], ["expr"])
add_primitive("set_x", ["op"], ["expr"])
add_primitive("set_y", ["op"], ["expr"])
add_primitive("change_angle", ["op"], ["expr"])
add_primitive("change_speed", ["op"], ["expr"])
add_primitive("move", ["op"], [])
add_primitive("dot", ["op"], [])
add_primitive("repeat", ["op"], ["op", "int_expr"])
add_primitive("concat", ["op"], ["op", "op"])
add_primitive("subprogram", ["op"], ["op"])
add_primitive("continue", ["op"], ["op"])
info["root"] = Dict("child_types"=>["op"])


function make_dist_dict()
    expr_dict = Dict()
    int_expr_dict = Dict()
    op_dict = Dict()
    op_to_op_c1_dict = Dict()
    expr_to_expr_c1_dict = Dict()
    op_to_op_c2_dict = Dict()
    expr_to_expr_c2_dict = Dict()

    for p in node_list
        p_dict = info[p]
        n_children = length(p_dict["child_types"])
        n_children_of_same_type = length([t for t in p_dict["child_types"] if t in p_dict["types"]])
        if "expr" in p_dict["types"]
            expr_dict[p] = 1 / 2^(n_children_of_same_type)
            if "int_expr" in p_dict["types"]
                int_expr_dict[p] = 1 / 2^(length([t for t in p_dict["child_types"] if t == "int_expr"]))
            end
            if n_children > 0
                if p_dict["child_types"][1] == "expr"
                    expr_to_expr_c1_dict[p] = 1 / 2^(n_children_of_same_type) 
                end
            end
            if n_children > 1
                if p_dict["child_types"][2] == "expr"
                    expr_to_expr_c2_dict[p] = 1 / 2^(n_children_of_same_type) 
                end
            end
        end
        if "op" in p_dict["types"]
            op_dict[p] = 1 / 2^(n_children_of_same_type)
            if n_children > 0
                if p_dict["child_types"][1] == "op"
                    op_to_op_c1_dict[p] = 1 / 2^(n_children_of_same_type) 
                end
            end
            if n_children > 1
                if p_dict["child_types"][2] == "op"
                    op_to_op_c2_dict[p] = 1 / 2^(n_children_of_same_type) 
                end
            end
        end
    end
    dist_dict = Dict(
        "expr" => dict_to_dist(expr_dict, node_list),
        "op" => dict_to_dist(op_dict, node_list),
        "op_to_op_c1" => dict_to_dist(op_to_op_c1_dict, node_list),
        "expr_to_expr_c1" => dict_to_dist(expr_to_expr_c1_dict, node_list),
        "op_to_op_c2" => dict_to_dist(op_to_op_c2_dict, node_list),
        "expr_to_expr_c2" => dict_to_dist(expr_to_expr_c2_dict, node_list),
        "int_expr" => dict_to_dist(int_expr_dict, node_list)
    )
    return dist_dict
end

dist_dict = make_dist_dict()


function get_node_dist_swap(change_node)
    #we need a node that is an acceptable child of change_node's parent, 
    #and an acceptable parent of change node's children
    prob_dict = Dict()
    for node_tp in node_list
        #we want to get a node that has changenode type as an acceptable c1 type  
        if length(info[node_tp]["child_types"])==length(info[change_node.tp]["child_types"])
            ok = true
            for (i,c) in enumerate(change_node.children)
                if ~(info[node_tp]["child_types"][i] in info[c.tp]["types"])
                    ok=false
                end
            end
            #and that is an acceptable c of changenode's parent
            if ok && info[change_node.parent_tp]["child_types"][change_node.parent_c] in info[node_tp]["types"]
                prob_dict[node_tp] = 1 #should make this relative to n_children?
            end
        end
    end
    return dict_to_dist(prob_dict, node_list)
end

function get_node_dist_add_c1(change_node)
    prob_dict = Dict()
    for node_tp in node_list
        #we want to get a node that has changenode type as an acceptable c1 type  
        if length(info[node_tp]["child_types"])>0 && info[node_tp]["child_types"][1] in info[change_node.tp]["types"]
            #and that is an acceptable c of changenode's parent
            if info[change_node.parent_tp]["child_types"][change_node.parent_c] in info[node_tp]["types"]
                prob_dict[node_tp] = 1 #should make this relative to n_children?
            end
        end
    end
    if sum(values(prob_dict))==0
        return nothing
    else
        return dict_to_dist(prob_dict, node_list)
    end
end

function get_node_dist_add_c2(change_node)
    prob_dict = Dict()
    for node_tp in node_list
        #we want to get a node that has changenode type as an acceptable c1 type  
        if length(info[node_tp]["child_types"])>1 && info[node_tp]["child_types"][2] in info[change_node.tp]["types"]
            #and that is an acceptable c of changenode's parent
            if info[change_node.parent_tp]["child_types"][change_node.parent_c] in info[node_tp]["types"]
                prob_dict[node_tp] = 1
            end
        end
    end
    if sum(values(prob_dict))==0
        return nothing
    else
        return dict_to_dist(prob_dict, node_list)
    end
end