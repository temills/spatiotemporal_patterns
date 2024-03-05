include("grammar.jl")

using Distributions
using Printf

function get_node_dist(cur::Int, changepoints::Bool, max_depth::Float64)
    level::Int = 1 + floor(log2(cur))
    @assert level <= max_depth
    if level == max_depth
        return node_dist_leaf
    elseif changepoints
        return node_dist_cp
    else
        return node_dist_nocp
    end
end

@gen function covariance_prior(node_type)

    if node_type == LINEAR
        intercept = @trace(uniform_continuous(0, 1), (cur, :intercept))
        dim = @trace(categorical(dim_dist), (cur, :dim))
        node = Linear(intercept, dim)

    elseif node_type == SQUARED_EXP
        scale = @trace(uniform_continuous(0, 1), (cur, :scale))
        dim = @trace(categorical(dim_dist), (cur, :dim))
        node = SquaredExponential(scale, dim)

    elseif node_type == PERIODIC
        scale = @trace(uniform_continuous(0, 1), (cur, :scale))
        period = @trace(uniform_continuous(0, 1), (cur, :period))
        dim = @trace(categorical(dim_dist), (cur, :dim))
        node = Periodic(scale, period, dim)

    elseif node_type == RATIONAL_QUADRATIC

    else
        error("Unknown node type: $node_type")
    end

    return node
end

MAX_NOISE = .1 

#here the mvnormal, based on the cov fn and noise and input vector, defines an output vector
@gen function model(input::Vector{Vector{Float64}}, node_type)
    n = length(input[1])
    covariance_fn_x = @trace(covariance_prior(node_type), :tree_x)
    covariance_fn_y = @trace(covariance_prior(node_type), :tree_y)
    noise_x = @trace(uniform(.001, MAX_NOISE), :noise_x)
    noise_y = @trace(uniform(.001, MAX_NOISE), :noise_y)

    cov_matrix_x = compute_cov_matrix_vectorized(covariance_fn_x, noise_x, input)
    cov_matrix_y = compute_cov_matrix_vectorized(covariance_fn_y, noise_y, input)
    @trace(mvnormal(zeros(n), cov_matrix_x), :output_x)
    @trace(mvnormal(zeros(n), cov_matrix_y), :output_y)

    return covariance_fn_x, covariance_fn_y
end

@gen function noise_proposal_x(prev_trace)
    return @trace(uniform(.001, MAX_NOISE), :noise_x)
end
@gen function noise_proposal_y(prev_trace)
    return @trace(uniform(.001, MAX_NOISE), :noise_y)
end

@gen function hyper_proposal(prev_trace, addr)
    prev_val = prev_trace[addr]
    input_left = max(0, prev_val - .1)
    input_right = min(1, prev_val + .1)
    prop_val = @trace(uniform_continuous(input_left, input_right), addr)
    return prop_val
end

#here we need to scale/shift mean by approp amount and scale var
# Computing predictions on held-out data for single particle.
function compute_particle_predictions(
        trace,
        input_train::Vector{Vector{Float64}},
        output_train_x::Vector{Float64},
        output_train_y::Vector{Float64},
        input_test::Vector{Vector{Float64}},
        output_test_x::Vector{Float64},
        output_test_y::Vector{Float64})
    # Obtain covariance and noise.
    cov_fn_x, cov_fn_y = trace[]
    noise_x = trace[:noise_x]
    noise_y = trace[:noise_y]

    # Compute probe points for interpolation and extrapolation.
    input_probe = [vcat(input_train[d], input_test[d]) for d in 1:length(input_train)]
 
    # Compute posterior predictive.
    (output_probe_mean_x, output_probe_cov_x) = compute_predictive(
        cov_fn_x, noise_x, input_train, output_train_x, input_probe)
    (output_probe_mean_y, output_probe_cov_y) = compute_predictive(
        cov_fn_y, noise_y, input_train, output_train_y, input_probe)
    

    output_probe_var_x = []
    for i=1:length(output_probe_mean_x)
         push!(output_probe_var_x, output_probe_cov_x[i,i])
    end
    output_probe_var_y = []
    for i=1:length(output_probe_mean_y)
         push!(output_probe_var_y, output_probe_cov_y[i,i])
    end

    # Return the results.
    return Dict(
        "cov_fn_x"           => cov_fn_x,
        "noise_x"            => noise_x,
        "input_probe"         => input_probe,
        "output_probe_mean_x"    => output_probe_mean_x,
        "output_probe_var_x"    => output_probe_var_x,
        "cov_fn_x"           => cov_fn_x,
        "cov_fn_y"           => cov_fn_y,
        "noise_y"            => noise_y,
        "output_probe_mean_y"    => output_probe_mean_y,
        "output_probe_var_y"    => output_probe_var_y,
    )
end


# Custom implementation of particle_filter_step! in Gen
# that removes the error on !isempty(discard)
function Gen.particle_filter_step!(
        state::Gen.ParticleFilterState{U},
        new_args::Tuple,
        argdiffs::Tuple,
        observations::ChoiceMap) where {U}
    num_particles = length(state.traces)
    Threads.@threads for i=1:num_particles
        (state.new_traces[i], incr, _, discard) = update(
            state.traces[i], new_args, argdiffs, observations)
        state.log_weights[i] += incr
    end
    # swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return nothing
end

function experiment_smc(
        input::Vector{Vector{Float64}},
        output_x::Vector{Float64},
        output_y::Vector{Float64},
        seq_id,
        n_particles::Int,
        n_mcmc,
        fname,
        kernel_type;
        biased::Bool=false,
        t_start::Int=1,
        t_step::Int=1,
        seed::Int=1,
        save::Bool=true)

    # Initialize particle filter.
    input_train = [input[d][1:t_start] for d in 1:length(input)]
    output_train_x = output_x[1:t_start]
    output_train_y = output_y[1:t_start]
    state = initialize_particle_filter(
                model,
                (input_train, kernel_type),
                choicemap((:output_x, output_train_x), (:output_y, output_train_y)),
                n_particles)
    argdiff = UnknownChange()

    prediction_dict = Dict("tpt"=> [], "particle"=> [], "weight"=> [], "true_x"=> [], "true_y"=> [], "pred_x"=> [], "pred_y"=> [], "var_x"=> [], "var_y"=> [], "seq_id"=>[], "fn_x"=>[],  "fn_y"=>[], "noise_x"=>[], "noise_y"=>[], "score"=>[], "prev_x"=>[], "prev_y"=>[], "sd_x"=>[], "sd_y"=>[])
    #for tpt 0, want first true x/y, everything else missing
    for i=1:n_particles
        append!(prediction_dict["particle"], i)
        push!(prediction_dict["weight"], missing)
        append!(prediction_dict["tpt"], 0)
        append!(prediction_dict["true_x"], input[2][1])
        append!(prediction_dict["true_y"], input[3][1])
        push!(prediction_dict["seq_id"], seq_id)
        push!(prediction_dict["fn_x"], missing)
        push!(prediction_dict["fn_y"], missing)
        push!(prediction_dict["noise_x"], missing)
        push!(prediction_dict["noise_y"], missing)
        push!(prediction_dict["score"], missing)
        push!(prediction_dict["pred_x"], missing)
        push!(prediction_dict["pred_y"], missing)
        push!(prediction_dict["prev_x"],  missing)
        push!(prediction_dict["prev_y"], missing)
        push!(prediction_dict["var_x"], missing)
        push!(prediction_dict["var_y"], missing)
        push!(prediction_dict["sd_x"], missing)
        push!(prediction_dict["sd_y"], missing)
    end
    #for tpt 1, we want second true x/y, and first true x/y as prev x
    #this will be our first input point, for a prediction at tpt 2
    for i=1:n_particles
        append!(prediction_dict["particle"], i)
        push!(prediction_dict["weight"], missing)
        append!(prediction_dict["tpt"], 1)
        append!(prediction_dict["true_x"], output_x[1])
        append!(prediction_dict["true_y"], output_y[1])
        push!(prediction_dict["seq_id"], seq_id)
        push!(prediction_dict["fn_x"], missing)
        push!(prediction_dict["fn_y"], missing)
        push!(prediction_dict["noise_x"], missing)
        push!(prediction_dict["noise_y"], missing)
        push!(prediction_dict["score"], missing)
        push!(prediction_dict["pred_x"], missing)
        push!(prediction_dict["pred_y"], missing)
        push!(prediction_dict["prev_x"], input[2][1])
        push!(prediction_dict["prev_y"], input[3][1])
        push!(prediction_dict["var_x"], missing)
        push!(prediction_dict["var_y"], missing)
        push!(prediction_dict["sd_x"], missing)
        push!(prediction_dict["sd_y"], missing)
    end


    # Sequentialize over observations.
    for t=t_start:t_step:length(input[1])-1
        println("Running SMC round $(t)")

        # Obtain dataset.
        input_train = [input[d][1:t] for d in 1:length(input)] 
        output_train_x = output_x[1:t]
        output_train_y = output_y[1:t]
        input_test = [input[d][t+1:t+1] for d in 1:length(input)]
        output_test_x = output_x[t+1:t+1]
        output_test_y = output_y[t+1:t+1]

        # Run particle filter step.
        Gen.particle_filter_step!(
                state,
                (input_train, changepoints, max_depth, min_loc, max_loc),
                (argdiff,),
                choicemap((:output_x, output_train_x), (:output_y, output_train_y)))

        # Resample if needed.
        #Gen.maybe_resample!(state, ess_threshold=n_particles/2)

        # Apply MCMC rejuvination to each particle.
        Threads.@threads for i=1:n_particles
            local tr = state.traces[i]
            for iter=1:n_mcmc
                # mh move on leaf addrs 
                #in lot, pass tup, in each tup list of symbols + number
                choices = Gen.get_choices(tr)
                submap_x = Gen.get_submap(choices, :tree_x)
                for (a, v) in Gen.get_values_shallow(submap_x)
                    if ~(a[2] in [:dim])
                        tr, accepted = metropolis_hastings(tr, hyper_proposal, (:tree_x=>a,))
                    end
                end
                submap_y = Gen.get_submap(choices, :tree_y)
                for (a, v) in Gen.get_values_shallow(submap_y)
                    if ~(a[2] in [:dim])
                        tr, accepted = metropolis_hastings(tr, hyper_proposal, (:tree_y=>a,))
                    end
                end
                # MH move on top-level white noise.
                for iter=1:5
                    tr, accepted = metropolis_hastings(tr, noise_proposal_x, ())
                    tr, accepted = metropolis_hastings(tr, noise_proposal_y, ())
                end
            end
            state.traces[i] = tr
        end
        # Compute predictions for each particle and save to disk
        n_particles = length(state.traces)
        t = length(input_train[1])
        #figure out if t is correct here - should be t+1?
        for i=1:n_particles
            predictions = compute_particle_predictions(state.traces[i], input_train, output_train_x, output_train_y, input_test, output_test_x, output_test_y)                
            append!(prediction_dict["particle"], i)
            push!(prediction_dict["weight"], state.log_weights[i])
            append!(prediction_dict["tpt"], t+1)
            append!(prediction_dict["true_x"], output_test_x[1])
            append!(prediction_dict["true_y"], output_test_y[1])
            append!(prediction_dict["pred_x"], predictions["output_probe_mean_x"][end])
            append!(prediction_dict["pred_y"], predictions["output_probe_mean_y"][end])
            push!(prediction_dict["prev_x"], output_train_x[end])
            push!(prediction_dict["prev_y"], output_train_y[end])
            append!(prediction_dict["var_x"], predictions["output_probe_var_x"][end])
            append!(prediction_dict["var_y"], predictions["output_probe_var_y"][end])
            append!(prediction_dict["sd_x"], predictions["output_probe_var_x"][end]^0.5)
            append!(prediction_dict["sd_y"], predictions["output_probe_var_y"][end]^0.5)
            push!(prediction_dict["seq_id"], seq_id)
            push!(prediction_dict["fn_x"], predictions["cov_fn_x"])
            push!(prediction_dict["fn_y"], predictions["cov_fn_y"])
            push!(prediction_dict["noise_x"], predictions["noise_x"])
            push!(prediction_dict["noise_y"], predictions["noise_y"])
            append!(prediction_dict["score"], get_score(state.traces[i]))
        end
        df = DataFrame(prediction_dict)
        CSV.write(fname, df)
    end
    # Return final traces.
    return prediction_dict
end
