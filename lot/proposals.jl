
""" This file contains the code for making proposals during mcmc """

######## proposals for params ########

@dist prop_dist(x) = exp(normal(log(x), 0.1))
@gen function noise_x_proposal(trace)
    shape_x ~ prop_dist(trace[:shape_x])
    scale_x ~ prop_dist(trace[:scale_x])
    precision_x ~ prop_dist(trace[:precision_x])
end
@gen function noise_y_proposal(trace)
    shape_y ~ prop_dist(trace[:shape_y])
    scale_y ~ prop_dist(trace[:scale_y])
    precision_y ~ prop_dist(trace[:precision_y])
end

@dist angle_proposal_dist(x) = normal(x, 0.1)
@gen function init_angle_proposal(trace)
    init_angle ~ angle_proposal_dist(trace[:init_angle])
end

@dist speed_proposal_dist(x) = exp(normal(log(x), 0.25))
@gen function init_speed_proposal(trace)
    init_speed ~ speed_proposal_dist(trace[:init_speed])
end

@dist number_proposal_dist(x) = normal(x, 0.1)

#prob of being +- 1
@dist int_proposal_dist(x) = categorical(normalize(append!(fill(0.0, max(0,x-2)), [1,0,1])))
@gen function number_proposal(trace, addr_temp::Vector{Symbol}, param)
    addr = addr_temp[length(addr_temp)]
    is_int = (addr==:int_param)
    for i in length(addr_temp)-1:-1:1
        addr = addr_temp[i] => addr
    end
    if is_int
        @trace(int_proposal_dist(param), addr)
    else
        @trace(number_proposal_dist(param), addr)
    end
end

#get addrs of params for number proposals
function get_param_addr_list(node::Node, addr_so_far::Vector{Symbol}, addr_list::Vector{Tuple{Vector{Symbol}, Any}})
    if node.tp == "real"
        a = copy(addr_so_far)
        push!(addr_list, (push!(a, :param), node.params[1]))
    elseif node.tp == "int"
        a = copy(addr_so_far)
        push!(addr_list, (push!(a, :int_param), node.params[1]))
    elseif length(node.children)==1
        a = copy(addr_so_far)
        addr_list = get_param_addr_list(node.children[1], push!(a, :c1), addr_list)
    elseif length(node.children)==2
        a = copy(addr_so_far)
        b = copy(addr_so_far)
        addr_list = get_param_addr_list(node.children[1], push!(a, :c1), addr_list)
        addr_list = get_param_addr_list(node.children[2], push!(b, :c2), addr_list)
    end
    return addr_list
end


######## proposals on tree structure ########

@gen function random_node_path(node::Node, biased=true)
    if biased
        p_stop = (length(node.children) == 0) ? 1.0 : 0.5
    else
        p_stop = (length(node.children) == 0) ? 1.0 : 1/node.size
    end
    stop = @trace(bernoulli(p_stop), :stop)
    if stop
        return (:tree, node)
    else
        if length(node.children)==1
            (next_node, direction) = (node.children[1], :c1)
        elseif length(node.children)==2
            p_c1 = size(node.children[1]) / (size(node) - 1)
            (next_node, direction) = @trace(bernoulli(p_c1), :dir) ? (node.children[1], :c1) : (node.children[2], :c2)
        else
            error("Not set up for nodes with > 2 children")
        end
        (rest_of_path, final_node) = @trace(random_node_path(next_node), :rest_of_path)

        if isa(rest_of_path, Pair)
            return (:tree => direction => rest_of_path[2], final_node)
        else
            return (:tree => direction, final_node)
        end
    end
end

# regenerate a random subtree
@gen function regen_random_subtree(prev_trace, joint)
    (subtree_addr, change_node) = @trace(random_node_path(get_retval(prev_trace)), :path)
    if "expr" in info[change_node.tp]["types"]
        #do we need this to be an int_expr?
        @trace(pcfg_prior("expr", change_node.parent_tp, change_node.parent_c), :new_subtree)
    else
        @assert "op" in info[change_node.tp]["types"]
        @trace(pcfg_prior("op", change_node.parent_tp, change_node.parent_c), :new_subtree)
    end

    if sum(joint)>0
        @trace(gen_joint_params(joint), :joint_params)
    end

    return subtree_addr
end

function subtree_involution(trace, fwd_assmt::ChoiceMap, subtree_addr, proposal_args::Tuple)
    model_assmt = get_choices(trace)
    bwd_assmt = choicemap()
    set_submap!(bwd_assmt, :path, get_submap(fwd_assmt, :path))
    set_submap!(bwd_assmt, :new_subtree, get_submap(model_assmt, subtree_addr))
    new_trace_update = choicemap()
    set_submap!(new_trace_update, subtree_addr, get_submap(fwd_assmt, :new_subtree))

    #update noise params
    (joint,) = proposal_args
    if sum(joint)>0
        (bwd_assmt, new_trace_update) = noise_involution_helper(model_assmt, bwd_assmt, fwd_assmt, new_trace_update, joint)
    end
    (new_trace, weight, _, _) = update(trace, get_args(trace), (NoChange(),), new_trace_update)

    (new_trace, weight, _, _) = update(trace, get_args(trace), (NoChange(),), new_trace_update)
    (new_trace, bwd_assmt, weight)
end

# swap a node with a new node

@gen function swap_node(prev_trace, joint)
    #path address now contains all random choices made in random node path unbiased
    (subtree_addr, change_node) = @trace(random_node_path(get_retval(prev_trace), false), :path)
    node_dist = get_node_dist_swap(change_node)
    @trace(categorical(node_dist), :new_node_type)

    if sum(joint)>0
        @trace(gen_joint_params(joint), :joint_params)
    end

    return subtree_addr
end

function swap_node_involution(trace, fwd_assmt::ChoiceMap, subtree_addr, proposal_args::Tuple)
    model_assmt = get_choices(trace)
    bwd_assmt = choicemap()
    set_submap!(bwd_assmt, :path, get_submap(fwd_assmt, :path))
    bwd_assmt[:new_node_type] = model_assmt[subtree_addr=>:type]
    
    symb_list = Vector{Symbol}()
    temp_addr = subtree_addr
    while isa(temp_addr, Pair)
        push!(symb_list, temp_addr[1])
        temp_addr = temp_addr[2]
    end
    push!(symb_list, temp_addr)
    push!(symb_list, :type)
    addr = symb_list[length(symb_list)]
    for i in length(symb_list)-1:-1:1
        addr = symb_list[i] => addr
    end
    new_trace_update = choicemap()
    new_trace_update[addr] = fwd_assmt[:new_node_type]

    #update noise params
    (joint,) = proposal_args
    if sum(joint)>0
        (bwd_assmt, new_trace_update) = noise_involution_helper(model_assmt, bwd_assmt, fwd_assmt, new_trace_update, joint)
    end
    (new_trace, weight, _, _) = update(trace, get_args(trace), (NoChange(),), new_trace_update)

    (new_trace, bwd_assmt, weight)
end



######## add or remove a node ########
# for add:
# so, randomly choose a node, and then add a new parent node above it 
# the chosen node will be its first child
# if this new parent node needs a second child, we'll generate one

#for remove:
# randomly choose a node, then remove it and replace it with its first child
# this is only possible when the chosen node's first child is an acceptable child of its parent


@gen function get_node_no_c1(node_dist)
    t = @trace(categorical(node_dist), :type)
    node_type = node_list[t]
    child_types = info[node_type]["child_types"]
    if node_type == "real"
        @trace(number_prior_dist(), :param)
    elseif node_type == "int"
        @trace(int_prior_dist(), :int_param)
    end
    if length(child_types) > 1
        @trace(pcfg_prior(child_types[2], node_type, 2), :c2)
    end
end


function get_can_remove_c1(change_node)
    if length(change_node.children) == 0
        return false
    end
    if change_node.parent_tp == "root"
        return "op" in info[change_node.children[1].tp]["types"]
    else
        most_general_ok_type = info[change_node.parent_tp]["child_types"][change_node.parent_c]
        #if the child can be classified as this type, we can replace the node with its child
        return most_general_ok_type in info[change_node.children[1].tp]["types"]
    end
end

#only follow paths that can lead to removable node paths
@gen function random_node_path_for_remove_c1(node::Node, path_so_far, paths_to_removable)
    if length(node.children)==0
        p_stop = 1
    elseif get_can_remove_c1(node)
        p_stop = 0.5
    else
        p_stop = 0
    end
    if @trace(bernoulli(p_stop), :stop)
        return (:tree, node)
    else
        if length(node.children)==1
            (next_node, direction) = (node.children[1], :c1)
        elseif length(node.children)==2
            p_c1 = size(node.children[1]) / (size(node) - 1)
            (next_node, direction) = @trace(bernoulli(p_c1), :dir) ? (node.children[1], :c1) : (node.children[2], :c2)
        else
            error("Not set up for nodes with > 2 children")
        end
        (rest_of_path, final_node) = @trace(random_node_path_for_remove_c1(next_node), :rest_of_path)

        if isa(rest_of_path, Pair)
            return (:tree => direction => rest_of_path[2], final_node)
        else
            return (:tree => direction, final_node)
        end
    end
end

@gen function add_or_remove_c1(prev_trace, joint)
    (subtree_addr, change_node) = @trace(random_node_path(get_retval(prev_trace)), :path)
    if get_can_remove_c1(change_node)
        add_prob = 0.3
    else
        add_prob = 1
    end
    add ~ bernoulli(add_prob)
    #if we are adding a parent to change_node,
    #we want to get a node that has changenode type as an acceptable child type
    #and that is an acceptable c1 of changenode's parent
    if add
        node_dist = get_node_dist_add_c1(change_node)
        @trace(get_node_no_c1(node_dist), :new_subtree)
    end

    if sum(joint)>0
        @trace(gen_joint_params(joint), :joint_params)
    end

    return subtree_addr
end


#add a node above, or remove the given node and replace w child
function add_or_remove_involution_c1(trace, fwd_assmt::ChoiceMap, subtree_addr, proposal_args::Tuple)
    model_assmt = get_choices(trace)
    bwd_assmt = choicemap()
    set_submap!(bwd_assmt, :path, get_submap(fwd_assmt, :path))
    bwd_assmt[:add] = ~(fwd_assmt[:add])

    #add node
    if fwd_assmt[:add]
        my_map = choicemap()
        #get type from new_subtree
        my_map[:type] = get_value(get_submap(fwd_assmt, :new_subtree), :type)
        #get c2 from new_subtree
        c2_map = get_submap(fwd_assmt, :new_subtree => :c2)
        if ~isempty(c2_map)
            set_submap!(my_map, :c2, c2_map)
        end
        #set c1 to the changenode
        set_submap!(my_map, :c1, get_submap(model_assmt, subtree_addr))
        new_trace_update = choicemap()
        #put this type/c1/c2 at the subtree addt in new_trace_update
        set_submap!(new_trace_update, subtree_addr, my_map)
    else
        c1_map = get_submap(model_assmt, subtree_addr=>:c1)
        new_trace_update = choicemap()
        set_submap!(new_trace_update, subtree_addr, c1_map)
        
        my_map = choicemap()
        my_map[:type] = get_value(model_assmt, subtree_addr => :type)
        c2_map = get_submap(model_assmt, subtree_addr => :c2)
        if ~isempty(c2_map)
            set_submap!(my_map, :c2, c2_map)
        end
        set_submap!(bwd_assmt, :new_subtree, my_map)
    end

    #update noise params
    (joint,) = proposal_args
    if sum(joint)>0
        (bwd_assmt, new_trace_update) = noise_involution_helper(model_assmt, bwd_assmt, fwd_assmt, new_trace_update, joint)
    end
    (new_trace, weight, _, _) = update(trace, get_args(trace), (NoChange(),), new_trace_update)

    (new_trace, bwd_assmt, weight)
end


######## add or remove a node, c2 version ########
# for add:
# so, randomly choose a node, and then add a new parent node above it 
# the chosen node will be its SECOND child
# generate a first child for the parent node

# for remove:
# randomly choose a node, then remove it and replace it with its SECOND child
# this is only possible when the chosen node's second child is an acceptable child of its parent

# all the code below is analogous to the c1 version above
@gen function get_node_no_c2(node_dist)
    t = @trace(categorical(node_dist), :type)
    node_type = node_list[t]
    child_types = info[node_type]["child_types"]
    if node_type == "real"
        @trace(number_prior_dist(), :param)
    elseif node_type == "int"
        @trace(int_prior_dist(), :int_param)
    end
    if length(child_types) > 0
        @trace(pcfg_prior(child_types[1], node_type, 1), :c1)
    end
end

function get_can_remove_c2(change_node)
    if length(change_node.children) < 2
        return false
    end
    if change_node.parent_tp == "root"
        return "op" in info[change_node.children[2].tp]["types"]
    else
        most_general_ok_type = info[change_node.parent_tp]["child_types"][change_node.parent_c]
        #if the child can be classified as this type, we can replace the node with its child
        return most_general_ok_type in info[change_node.children[2].tp]["types"]
    end
end

@gen function random_node_path_for_remove_c2(node::Node)
    if length(node.children)==0
        p_stop = 1
    elseif get_can_remove_c2(node)
        p_stop = 0.5
    else
        p_stop = 0
    end

    if @trace(bernoulli(p_stop), :stop)
        return (:tree, node)
    else
        if length(node.children)==1
            (next_node, direction) = (node.children[1], :c1)
        elseif length(node.children)==2
            p_c1 = size(node.children[1]) / (size(node) - 1)
            (next_node, direction) = @trace(bernoulli(p_c1), :dir) ? (node.children[1], :c1) : (node.children[2], :c2)
        else
            error("Not set up for nodes with > 2 children")
        end
        (rest_of_path, final_node) = {:rest_of_path} ~ random_node_path_for_remove_c2(next_node)

        if isa(rest_of_path, Pair)
            return (:tree => direction => rest_of_path[2], final_node)
        else
            return (:tree => direction, final_node)
        end
    end
end


@gen function add_or_remove_c2(prev_trace, joint)
    (subtree_addr, change_node) = @trace(random_node_path(get_retval(prev_trace)), :path)
    if get_can_remove_c2(change_node)
        add_prob = 0.3
    else
        add_prob = 1
    end
    add ~ bernoulli(add_prob)
    if add
        node_dist = get_node_dist_add_c2(change_node)
        @trace(get_node_no_c2(node_dist), :new_subtree)
    end

    if sum(joint)>0
        @trace(gen_joint_params(joint), :joint_params)
    end

    return subtree_addr
end


function add_or_remove_involution_c2(trace, fwd_assmt::ChoiceMap, subtree_addr, proposal_args::Tuple)
    model_assmt = get_choices(trace)
    bwd_assmt = choicemap()
    set_submap!(bwd_assmt, :path, get_submap(fwd_assmt, :path))
    bwd_assmt[:add] = ~(fwd_assmt[:add])

    #add node
    if fwd_assmt[:add]
        my_map = choicemap()
        my_map[:type] = get_value(get_submap(fwd_assmt, :new_subtree), :type)
        set_submap!(my_map, :c2, get_submap(model_assmt, subtree_addr))
        set_submap!(my_map, :c1, get_submap(fwd_assmt, :new_subtree => :c1))
        new_trace_update = choicemap()
        set_submap!(new_trace_update, subtree_addr, my_map)
    #remove node
    else
        c2_map = get_submap(model_assmt, subtree_addr=>:c2)
        new_trace_update = choicemap()
        set_submap!(new_trace_update, subtree_addr, c2_map)
        my_map = choicemap()
        my_map[:type] = get_value(model_assmt, subtree_addr => :type)
        set_submap!(my_map, :c1, get_submap(model_assmt, subtree_addr => :c1))
        set_submap!(bwd_assmt, :new_subtree, my_map)
    end

    #update noise params
    (joint,) = proposal_args
    if sum(joint)>0
        (bwd_assmt, new_trace_update) = noise_involution_helper(model_assmt, bwd_assmt, fwd_assmt, new_trace_update, joint)
    end
    (new_trace, weight, _, _) = update(trace, get_args(trace), (NoChange(),), new_trace_update)

    (new_trace, bwd_assmt, weight)
end


@gen function gen_joint_params(joint)
    if joint[1]==1
        new_shape_x ~ gamma(10,10)
        new_scale_x ~ gamma(1,1)
        new_precision_x ~ gamma(new_shape_x,new_scale_x)
        new_shape_y ~ gamma(10,10)
        new_scale_y ~ gamma(1,1)
        new_precision_y ~ gamma(new_shape_y,new_scale_y)
    end
    if joint[2]==1
        new_init_angle ~ uniform(-4,4)
    end
    if joint[3]==1
        new_init_speed ~ exponential(0.5)
    end
end

function noise_involution_helper(model_assmt, bwd_assmt, fwd_assmt, new_trace_update, joint)
    if joint[1]==1
        bwd_assmt[:joint_params=>:new_shape_x] = model_assmt[:shape_x]
        bwd_assmt[:joint_params=>:new_scale_x] = model_assmt[:scale_x]
        bwd_assmt[:joint_params=>:new_precision_x] = model_assmt[:precision_x]
        bwd_assmt[:joint_params=>:new_shape_y] = model_assmt[:shape_y]
        bwd_assmt[:joint_params=>:new_scale_y] = model_assmt[:scale_y]
        bwd_assmt[:joint_params=>:new_precision_y] = model_assmt[:precision_y]
        #now set new trace based on fwd assmt
        new_trace_update[:shape_x] = fwd_assmt[:joint_params=>:new_shape_x]
        new_trace_update[:scale_x] = fwd_assmt[:joint_params=>:new_scale_x]
        new_trace_update[:precision_x] = fwd_assmt[:joint_params=>:new_precision_x]
        new_trace_update[:shape_y] = fwd_assmt[:joint_params=>:new_shape_y]
        new_trace_update[:scale_y] = fwd_assmt[:joint_params=>:new_scale_y]
        new_trace_update[:precision_y] = fwd_assmt[:joint_params=>:new_precision_y]
    end
    if joint[2]==1
        bwd_assmt[:joint_params=>:new_init_angle] = model_assmt[:init_angle]
        new_trace_update[:init_angle] = fwd_assmt[:joint_params=>:new_init_angle]
    end
    if joint[3]==1
        bwd_assmt[:joint_params=>:new_init_speed] = model_assmt[:init_speed]
        new_trace_update[:init_speed] = fwd_assmt[:joint_params=>:new_init_speed]
    end

    return (bwd_assmt, new_trace_update)
end