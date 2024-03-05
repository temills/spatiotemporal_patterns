using Gen
import LinearAlgebra
import Random

using Statistics: mean
using LinearAlgebra: Symmetric

################################
# Abstract Covariance Function #
################################

"""
Node in a tree representing a covariance function.
"""
abstract type Node end
abstract type LeafNode <: Node end
abstract type BinaryOpNode <: Node end

"""
Number of nodes in the subtree rooted at this node.
"""
Base.size(::LeafNode) = 1
Base.size(node::BinaryOpNode) = node.size


"""
Constant kernel
"""
struct Constant <: LeafNode
    value::Real
    size::Int
end
Constant(v) = Constant(v, 1)
eval_cov(node::Constant, input_pairs::Vector{Tuple{Float64, Float64}}) = node.value

function eval_cov_mat(node::Constant, input::Vector{Vector{Float64}})
    n = length(input[1])
    return node.value .* LinearAlgebra.ones(n, n)
end

"""
Linear kernel
"""
struct Linear <: LeafNode
    intercept::Real
    dim::Int64
    size::Int
end
Linear(v1, v2) = Linear(v1, v2, 1)
function eval_cov(node::Linear, input_pairs::Vector{Tuple{Float64, Float64}})
    v1 = input_pairs[node.dim][1]
    v2 = input_pairs[node.dim][2]
    (v1 - node.intercept) * (v2 - node.intercept)
end

function eval_cov_mat(node::Linear, input::Vector{Vector{Float64}})
    vs = input[node.dim]
    vs_minus_intercept = reshape(vs, (length(vs), 1)) .- node.intercept
    return vs_minus_intercept * transpose(vs_minus_intercept)
end

"""
Squared exponential kernel
"""
struct SquaredExponential <: LeafNode
    scale::Real
    dim::Int64
    size::Int
end
SquaredExponential(v1, v2) = SquaredExponential(v1, v2, 1)
function eval_cov(node::SquaredExponential, input_pairs::Vector{Tuple{Float64, Float64}})
    v1 = input_pairs[node.dim][1]
    v2 = input_pairs[node.dim][2]
    return exp(-.5 * (v1 - v2) * (v1 - v2) / node.scale)
end

function eval_cov_mat(node::SquaredExponential, input::Vector{Vector{Float64}})
    vs = input[node.dim]
    dv = vs .- vs'
    return exp.(-.5 .* dv .* dv ./ node.scale)
end

"""
Periodic kernel
"""
struct Periodic <: LeafNode
    scale::Real
    period::Real
    dim::Int64
    size::Int
end
Periodic(v1, v2, v3) = Periodic(v1, v2, v3, 1)
function eval_cov(node::Periodic, input_pairs::Vector{Tuple{Float64, Float64}})
    v1 = input_pairs[node.dim][1]
    v2 = input_pairs[node.dim][2]
    freq = 2 * pi / node.period
    return exp((-1/node.scale) * (sin(freq * abs(v1 - v2)))^2)
end

function eval_cov_mat(node::Periodic, input::Vector{Vector{Float64}})
    vs = input[node.dim]
    freq = 2 * pi / node.period
    dv = abs.(vs .- vs')
    return exp.((-1/node.scale) .* (sin.(freq .* dv)).^2)
end

"""
Plus node
"""
struct Plus <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end

Plus(left, right) = Plus(left, right, 1 + size(left) + size(right))

function eval_cov(node::Plus, input_pairs::Vector{Tuple{Float64, Float64}})
    return eval_cov(node.left, input_pairs) + eval_cov(node.right, input_pairs)
end

function eval_cov_mat(node::Plus, input::Vector{Vector{Float64}})
    return eval_cov_mat(node.left, input) .+ eval_cov_mat(node.right, input)
end
 
"""
Times node
"""
struct Times <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end

Times(left, right) = Times(left, right, 1 + size(left) + size(right))

function eval_cov(node::Times, input_pairs::Vector{Tuple{Float64, Float64}})
    return eval_cov(node.left, input_pairs) * eval_cov(node.right, input_pairs)
end

function eval_cov_mat(node::Times, input::Vector{Vector{Float64}})
    return eval_cov_mat(node.left, input) .* eval_cov_mat(node.right, input)
end

"""
Change Point node
"""
struct ChangePoint <: BinaryOpNode
    left::Node
    right::Node
    location::Real
    scale::Real
    dim::Int64
    size::Int
end

ChangePoint(left, right, location, scale, dim) =
    ChangePoint(left, right, location, scale, dim, 1 + size(left) + size(right))

function sigma_cp(v::Float64, location, scale)
    return .5 * (1 + tanh((location - v) / scale))
end

function eval_cov(node::ChangePoint, input_pairs::Vector{Tuple{Float64, Float64}})
    v1 = input_pairs[node.dim][1]
    v2 = input_pairs[node.dim][2]
    sigma_v1 = sigma_cp(v1, node.location, node.scale)
    sigma_v2 = sigma_cp(v2, node.location, node.scale)
    k_left = sigma_v1 * eval_cov(node.left, input_pairs) * sigma_v2
    k_right = (1 - sigma_v1) * eval_cov(node.right, input_pairs) * (1 - sigma_v2)
    return k_left + k_right
end

function eval_cov_mat(node::ChangePoint, input::Vector{Vector{Float64}})
    vs = input[node.dim]
    change_v = sigma_cp.(vs, node.location, node.scale)
    sig_1 = change_v * change_v'
    sig_2 = (1 .- change_v) * (1 .- change_v')
    k_1 = eval_cov_mat(node.left, input)
    k_2 = eval_cov_mat(node.right, input)
    return Symmetric(sig_1 .* k_1 + sig_2 .* k_2)
end

"""
Compute covariance matrix by evaluating the function on each pair of inputs.
"""
function compute_cov_matrix(covariance_fn::Node, noise, input)
    n = length(input[1])
    cov_matrix = Matrix{Float64}(undef, n, n)
    for i=1:n
        for j=1:n
            input_pairs = [(input[d][i], input[d][j]) for d=1:length(input)]
            cov_matrix[i, j] = eval_cov(covariance_fn, input_pairs)
        end
        cov_matrix[i, i] += noise
    end
    return cov_matrix
end

"""
Compute covariance function by recursively computing covariance matrices.
"""
function compute_cov_matrix_vectorized(covariance_fn, noise, input)
    n = length(input[1])
    return eval_cov_mat(covariance_fn, input) + Matrix(noise * LinearAlgebra.I, n, n)
end

function compute_log_likelihood(cov_matrix::Matrix{Float64}, output::Vector{Float64})
    n = length(output)
    return logpdf(mvnormal, output, zeros(n), cov_matrix)
end

"""
Return predictive log likelihood on new input values.
"""
function predictive_ll(
        covariance_fn::Node,
        noise::Float64,
        input::Vector{Vector{Float64}},
        output::Vector{Float64},
        new_input::Vector{Vector{Float64}},
        new_output::Vector{Float64})
    (conditional_mu, conditional_cov_matrix) = compute_predictive(
        covariance_fn, noise, input, output, new_input)
    return logpdf(mvnormal, new_output, conditional_mu, conditional_cov_matrix)
end

"""
Return mean and covariance of predictive distribution for new input values.
"""
function compute_predictive(
        covariance_fn::Node,
        noise::Float64,  
        input::Vector{Vector{Float64}},
        output::Vector{Float64},
        new_input::Vector{Vector{Float64}})
    n_prev = length(input[1])
    n_new = length(new_input[1])
    means = zeros(n_prev + n_new)
    cov_matrix = compute_cov_matrix(covariance_fn, noise, [vcat(input[d], new_input[d]) for d in 1:length(input)])
    cov_matrix_11 = cov_matrix[1:n_prev, 1:n_prev]
    cov_matrix_22 = cov_matrix[n_prev+1:n_prev+n_new, n_prev+1:n_prev+n_new]
    cov_matrix_12 = cov_matrix[1:n_prev, n_prev+1:n_prev+n_new]
    cov_matrix_21 = cov_matrix[n_prev+1:n_prev+n_new, 1:n_prev]
    # @assert cov_matrix_12 == cov_matrix_21'
    # CP kernel gives approximate symmetry due to tanh
    @assert isapprox(cov_matrix_12, cov_matrix_21')
    mu1 = means[1:n_prev]
    mu2 = means[n_prev+1:n_prev+n_new]
    conditional_mu = mu2 + cov_matrix_21 * (cov_matrix_11 \ (output - mu1))
    conditional_cov_matrix = cov_matrix_22 - cov_matrix_21 * (cov_matrix_11 \ cov_matrix_12)
    conditional_cov_matrix = .5 * conditional_cov_matrix + .5 * conditional_cov_matrix'
    return (conditional_mu, conditional_cov_matrix)
end

"""
Sample output values for new input values.
"""
function predict_output(covariance_fn::Node, noise::Float64,
        input::Vector{Vector{Float64}}, output::Vector{Float64}, new_input::Vector{Vector{Float64}})
    (conditional_mu, conditional_cov_matrix) = compute_predictive(
        covariance_fn, noise, input, output, new_input)
    return mvnormal(conditional_mu, conditional_cov_matrix)
end



"""
Compute mean squared error of predictions.
"""
function compute_mse(
        covariance_fn::Node,
        noise::Float64,
        input_train::Vector{Vector{Float64}},
        output_train::Vector{Float64},
        input_test::Vector{Vector{Float64}},
        output_test::Vector{Float64})
    (conditional_mu, _) = compute_predictive(
        covariance_fn, noise, input_train, output_train, input_test)
    return mean(sum((conditional_mu .- output_test) .^ 2))
end

"""
Mapping from kernel names to indexes.
"""
const CONSTANT    = 1
const LINEAR      = 2
const SQUARED_EXP = 3
const PERIODIC    = 4
const PLUS        = 5
const TIMES       = 6
const CHANGEPOINT = 7

"""
Distribution over kernel names.
"""
const node_dist_leaf = Float64[.25, .25, .25, .25]
const node_dist_nocp = Float64[.2, .2, .2, .2, .1, .1]
const node_dist_cp = Float64[.175, .175, .175, .175, .1, .1, .1]
const n_dims = 3
const dim_dist = [1/n_dims for i = 1:n_dims]

"""
Maximum number of children for any node.
"""
const max_branch = 2


"""
Select a node from the tree uniformly at random.
"""
@gen function pick_random_node(node::Node, cur::Int, depth::Int)
    # Choose a leaf.
    if isa(node, LeafNode)
        return (node, cur, depth)
    # Terminate w.p. 50%.
    elseif @trace(bernoulli(.5), :done => depth)
        return (node, cur, depth)
    # Recurse to left child.
    elseif @trace(bernoulli(.5), :recurse_left => cur)
        cur_next = Gen.get_child(cur, 1, max_branch)
        return @trace(pick_random_node(node.left, cur_next, depth + 1))
    # Recurse to right child.
    else
        cur_next = Gen.get_child(cur, 2, max_branch)
        return @trace(pick_random_node(node.right, cur_next, depth + 1))
    end
end

"""
Select a node from the tree, biased toward the root.
"""
@gen function pick_random_node_biased(node::Node, cur::Int, depth::Int)
    # Choose a leaf.
    if isa(node, LeafNode)
        return (node, cur, depth)
    end
    # Terminate w.p. p_done.
    p_done = 1 / size(node)
    if @trace(bernoulli(p_done), :done => depth)
        return (node, cur, depth)
    end
    p_recurse_left = size(node.left) / (size(node) - 1)
    # Recurse to left child.
    if @trace(bernoulli(p_recurse_left), :recurse_left => cur)
        cur_next = Gen.get_child(cur, 1, max_branch)
        return @trace(pick_random_node_biased(node.left, cur_next, depth + 1))
    # Recurse to right child.
    else
        cur_next = Gen.get_child(cur, 2, max_branch)
        return @trace(pick_random_node_biased(node.right, cur_next, depth + 1))
    end
end