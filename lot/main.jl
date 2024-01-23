include("grammar.jl")
include("utils.jl")
include("proposals.jl")
using CSV
using DataFrames
using Plots
using LinearAlgebra

""" This file contains the model and code to run smc """


@dist number_prior_dist() = normal(0, 1)
@dist function int_prior_dist()
    d = Vector{Float64}()
    for i in 1:20
        append!(d, 1-(i*.04))
    end
    d = normalize(d)
    categorical(d)
end
@gen function pcfg_prior(type_dist, parent_tp, parent_c)
    # draw from specified prior distribution
    t = @trace(categorical(dist_dict[type_dist]), :type)
    node_type = node_list[t]
    child_types = info[node_type]["child_types"]
    children = Vector{Node}()
    params = []
    if node_type == "real"
        param = @trace(number_prior_dist(), :param)
        push!(params, param)
    elseif node_type == "int"
        param = @trace(int_prior_dist(), :int_param)
        push!(params, param)
    end
    if length(child_types) > 0
        c1 = @trace(pcfg_prior(child_types[1], node_type, 1), :c1)
        push!(children, c1)
    end
    if length(child_types) > 1
        c2 = @trace(pcfg_prior(child_types[2], node_type, 2), :c2)
        push!(children, c2)
    end
    node = Node(node_type, parent_tp, parent_c, params, children)
    return node
end

@gen function model(n_to_predict::Integer, xs::Vector{Float64}, ys::Vector{Float64}, move_from_true=false)
    #sample function
    func::Node = @trace(pcfg_prior("op", "root", 1), :tree)
   
    #sample function params
    shape_x ~ gamma(10,10)
    scale_x ~ gamma(1,1)
    precision_x ~ gamma(shape_x,scale_x)
    shape_y ~ gamma(10,10)
    scale_y ~ gamma(1,1)
    precision_y ~ gamma(shape_y,scale_y)
    init_angle ~ uniform(-4,4)
    init_speed ~ exponential(0.5)
    
    wrap ~ bernoulli(0.5)

    out_x, out_y = evaluate_function(func, n_to_predict, xs, ys, init_speed, init_angle, move_from_true, wrap)
    
    for t in 1:n_to_predict
        ({(:x, t)} ~ normal(out_x[t], (1/precision_x)^0.5))
        ({(:y, t)} ~ normal(out_y[t], (1/precision_y)^0.5))
    end

    return func
end


@gen function evaluate_function(func::Node, n_to_predict::Integer, xs::Vector{Float64},
                            ys::Vector{Float64}, init_speed::Float64, init_angle, move_from_true::Bool, wrap::Bool)
    #initialize information to track as function is evaluated
    func_state = init_func_state(n_to_predict, xs, ys, init_speed, init_angle, move_from_true)
    #cutoff for func size to prevent very long evaluation times
    if func.size < 40
        try
            if wrap
                n1 = Node("concat", "increment_counter", 1, [], Vector{Node}([func, Node("move", "concat", 1, [], Vector{Node}())]))
            else
                n1=func
            end
            n2 = Node("increment_counter", "concat", 2, [], Vector{Node}())
            n3 = Vector{Node}([Node("concat", "continue", 1, [], Vector{Node}([n1, n2]))])
            node = Node("continue", "root", 1, [], n3)
            func_state = eval_node(node, func_state)
        catch err
            println(func)
            println(err)
        end
    end 
    while length(func_state["output_x"]) < n_to_predict
        if length(func_state["output_x"])==0
            push!(func_state["output_x"], xs[1])
            push!(func_state["output_y"], ys[1])
        else
            push!(func_state["output_x"], func_state["output_x"][end])
            push!(func_state["output_y"], func_state["output_y"][end])
        end
    end
    pred_x = func_state["output_x"][1:n_to_predict]
    pred_y = func_state["output_y"][1:n_to_predict]
    return pred_x, pred_y
end

# Run SMC with MCMC rejuvenation
function run_smc(
        xs::Vector{Float64},
        ys::Vector{Float64},
        n_particles::Integer,
        n_mcmc::Integer;
        seq_id="sequence",
        out_dir="output/",
        move_from_true=true,
        visualize=false)
    #init dict for storing predictions
    prediction_dict = init_pred_dict(xs, ys, seq_id, n_particles)
    #init particles with no observations
    observation = choicemap()
    state = initialize_particle_filter(
                model,
                (0, xs[1:1], ys[1:1], move_from_true),
                observation, n_particles)
    #record t0 predictions
    record_predictions(prediction_dict, state.traces, 0, xs, ys, seq_id, move_from_true, out_dir)
    #visualize
    fig=""
    if visualize
        fig = visualize_init(xs,ys)
    end
    #run SMC
    #obs at t -> predict at t+1
    for t=1:(length(xs)-2)
        #true xs
        observation = choicemap()
        observation[(:x, t)] = xs[t+1]
        observation[(:y, t)] = ys[t+1]
        # Run particle filter step on new observation, update weights
        Gen.particle_filter_step!(
                state,
                (t, xs[1:t+1], ys[1:t+1], move_from_true),
                (UnknownChange(), UnknownChange(), UnknownChange(), UnknownChange()),
                observation)
        do_resample = Gen.maybe_resample!(state, ess_threshold=n_particles/2)
        # apply MCMC rejuvenation to each particle
        #Threads.@threads for i=1:n_particles 
        for i=1:n_particles
            local trace = state.traces[i]
            vis_args = (t+1, xs[1:t+2], ys[1:t+2], fig, move_from_true)
            trace = mcmc_rejuvenation(trace, n_mcmc, vis_args, visualize)
            state.traces[i] = trace
        end
        #make predictions based sampled traces
        record_predictions(prediction_dict, state.traces, t, xs, ys, seq_id, move_from_true, out_dir)
    end
    #return state
    return prediction_dict
end

function mcmc_rejuvenation(trace, n_mcmc, vis_args, vis=false)
    acc_dict = Dict()
    for iter=1:n_mcmc
        #proposals on tree structure:
        joint = [bernoulli(1/4),bernoulli(1/4),bernoulli(1/4)]
        if ~(joint in collect(keys(acc_dict)))
            acc_dict[joint] = [0,0,0,0,0,0,0,0]
        end
        trace, a1 = mh(trace, regen_random_subtree, (joint,), subtree_involution) 
        trace, a2 = mh(trace, add_or_remove_c1, (joint,), add_or_remove_involution_c1)
        trace, a3 = mh(trace, add_or_remove_c2, (joint,), add_or_remove_involution_c2)
        trace, a4 = mh(trace, swap_node, (joint,), swap_node_involution)
        if a1
            acc_dict[joint][1] = acc_dict[joint][1]+1
        else
            acc_dict[joint][5] = acc_dict[joint][5]+1
        end
        if a2
            acc_dict[joint][2] = acc_dict[joint][2]+1
        else
            acc_dict[joint][6] = acc_dict[joint][6]+1
        end
        if a3
            acc_dict[joint][3] = acc_dict[joint][3]+1
        else
            acc_dict[joint][7] = acc_dict[joint][7]+1
        end
        if a4
            acc_dict[joint][4] = acc_dict[joint][4]+1
        else
            acc_dict[joint][8] = acc_dict[joint][8]+1
        end

        #proposals on numbers (ints/floats)
        addr_list = get_param_addr_list(trace[:tree], [:tree], Vector{Tuple{Vector{Symbol}, Any}}())
        for tup in addr_list
            trace, = mh(trace, number_proposal, tup)
        end
        #proposals on params
        for _=1:5
            trace, = mh(trace, init_angle_proposal, ())
            trace, = mh(trace, init_speed_proposal, ())
            trace, = mh(trace, noise_x_proposal, ())
            trace, = mh(trace, noise_x_proposal, ())
            trace, = mh(trace, Gen.select(:wrap))
        end
        if vis && (iter%1000)==0
            visualize_curr(vis_args, trace)
            println(acc_dict)
        end
    end
    return trace
end

function visualize_curr(vis_args, trace)
    (t, xs, ys, fig, move_from_true) = vis_args
    func = get_retval(trace)
    xs_model, ys_model = evaluate_function(func, t, xs[1:end-1], ys[1:end-1], trace[:init_speed], trace[:init_angle], move_from_true, trace[:wrap])
    fig = ""
    fig = visualize_init(xs,ys)
    gui(scatter!(fig, xs_model, ys_model, c="blue", label=nothing))
    println(node2str(func))
    println(get_score(trace))
    println(trace[:init_angle])
    println(trace[:init_speed])
end

function record_predictions(prediction_dict, traces, t, xs, ys, seq_id, move_from_true, out_dir)
    for (i, trace) in enumerate(traces)
        func = get_retval(trace)
        xs_model, ys_model = evaluate_function(func, t+1, xs[1:t+1], ys[1:t+1], trace[:init_speed], trace[:init_angle], move_from_true, trace[:wrap])   
        push!(prediction_dict["particle"], i)
        push!(prediction_dict["tpt"], t+1)
        push!(prediction_dict["seq_id"], seq_id)
        push!(prediction_dict["func"], node2str(func))
        push!(prediction_dict["init_angle"], trace[:init_angle])
        push!(prediction_dict["init_speed"], trace[:init_speed])
        push!(prediction_dict["true_x"], xs[t+2])
        push!(prediction_dict["true_y"], ys[t+2])
        push!(prediction_dict["pred_x"], xs_model[t+1])
        push!(prediction_dict["pred_y"], ys_model[t+1])
        push!(prediction_dict["shape_x"], trace[:shape_x])
        push!(prediction_dict["scale_x"], trace[:scale_x])
        push!(prediction_dict["precision_x"], trace[:precision_x])
        push!(prediction_dict["shape_y"], trace[:shape_y])
        push!(prediction_dict["scale_y"], trace[:scale_y])
        push!(prediction_dict["precision_y"], trace[:precision_y])
        push!(prediction_dict["score"], get_score(trace))
        push!(prediction_dict["wrap"], trace[:wrap])
        push!(prediction_dict["sd_x"], (1/trace[:precision_x])^0.5)
        push!(prediction_dict["sd_y"], (1/trace[:precision_y])^0.5)
    end
    df = DataFrame(prediction_dict)
    CSV.write(out_dir * seq_id * ".csv", df)
end


function init_pred_dict(xs, ys, seq_id, n_particles)
    prediction_dict = Dict("tpt"=> [], "particle"=> [], "true_x"=> [], "true_y"=> [], "pred_x"=> [], "pred_y"=>[], "seq_id"=>[], "func"=>[], "shape_x"=>[], "shape_y"=>[], "scale_x"=>[], "scale_y"=>[], "sd_x"=>[], "sd_y"=>[], "precision_x"=>[], "precision_y"=>[], "init_angle"=>[], "init_speed"=>[], "score"=>[], "wrap"=>[])
    for i=1:n_particles
        push!(prediction_dict["particle"], i)
        push!(prediction_dict["tpt"], 0)
        push!(prediction_dict["true_x"], xs[1])
        push!(prediction_dict["true_y"], ys[1])
        push!(prediction_dict["seq_id"], seq_id)
        push!(prediction_dict["func"], missing)
        push!(prediction_dict["shape_x"], missing)
        push!(prediction_dict["scale_x"], missing)
        push!(prediction_dict["precision_x"], missing)
        push!(prediction_dict["shape_y"], missing)
        push!(prediction_dict["scale_y"], missing)
        push!(prediction_dict["precision_y"], missing)
        push!(prediction_dict["init_angle"], missing)
        push!(prediction_dict["init_speed"], missing)
        push!(prediction_dict["score"], missing)
        push!(prediction_dict["pred_x"], missing)
        push!(prediction_dict["pred_y"], missing)
        push!(prediction_dict["sd_x"], missing)
        push!(prediction_dict["sd_y"], missing)
        push!(prediction_dict["wrap"], missing)
    end
    return prediction_dict
end