include("main.jl")

using CSV
using DataFrames
using JSON


function scale_pts(xs, ys)
    #scale so mean distance bt points is 1
    distances = [sqrt((xs[i+1] - xs[i])^2 + (ys[i+1] - ys[i])^2) for i in 1:length(xs)-1]
    mean_dist = mean(distances)
    xs = xs./mean_dist
    ys = ys./mean_dist
    return xs, ys
end

n_iter = 20000
n_particles = 1

vis = false
out_dir = "output/"

seq_dict = JSON.parsefile("stimuli.json")["funcs"]
seq_names = collect(keys(seq_dict))


for seq_name in seq_names
    println(seq_name)
    seq_info = seq_dict[seq_name]

    #ground truth
    xs = Vector{Float64}(seq_info["true_coords"][1])
    ys = Vector{Float64}(seq_info["true_coords"][2])
    xs, ys = scale_pts(xs, ys)
    # SMC Experiment.
    trace_dict = run_smc(xs, ys, n_particles, n_iter, seq_id=seq_name, out_dir=out_dir, move_from_true=true, visualize=true)
    df = DataFrame(trace_dict)
    CSV.write(out_dir * seq_name * ".csv", df)
end
