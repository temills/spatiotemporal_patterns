using CSV
using DataFrames
using JSON
using Profile

include("main.jl")

"""
function scale_pts(xs, ys)
    #mean distance bt points is 1
    distances = [sqrt((xs[i+1] - xs[i])^2 + (ys[i+1] - ys[i])^2) for i in 1:length(xs)-1]
    mean_dist = mean(distances)
    xs = [xs[i]/mean_dist for i=1:length(xs)]
    ys = [ys[i]/mean_dist for i=1:length(xs)]
    return xs, ys
end
"""

function scale_pts(l)
    #mean distance bt points is 1
    if minimum(l)==maximum(l)
        l = [0. for _=1:length(l)]
    else
        l = l.-minimum(l)
        l = l./maximum(l)
    end
    return l
end


n_iter = 100000
n_particles = 1
seq_dict = JSON.parsefile("/Users/traceymills/Dropbox (MIT)/cocosci_projects/dots/spatiotemporal/models/stimuli.json")["funcs"]
#seq_dict = JSON.parsefile("../stimuli.json")["funcs"]

outdir = "temp/"
seq_names = collect(keys(seq_dict))
for seq_name in seq_names
    println(seq_name)
    seq_name = "zigzag_increasing"
    seq_info = seq_dict[seq_name]

    xs = Vector{Float64}(seq_info["true_coords"][1])
    ys = Vector{Float64}(seq_info["true_coords"][2])
    xs = scale_pts(xs)
    ys = scale_pts(ys)
    ts = collect(1:(length(xs)-1))
    ts = scale_pts(ts)
    prev_xs = Vector{Float64}(xs[1:end-1])
    prev_ys = Vector{Float64}(ys[1:end-1])
    input = [ts, prev_xs, prev_ys] 
    out_x = xs[2:end]
    out_y = ys[2:end]

    # run SMC
    trace_dict = experiment_smc(input, out_x, out_y, seq_name, n_particles, n_iter, outdir * seq_name * ".csv"; changepoints=true)
    df = DataFrame(trace_dict)
    CSV.write(outdir * seq_name * ".csv", df)
    break
end
