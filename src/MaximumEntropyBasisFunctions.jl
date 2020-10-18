module MaximumEntropyBasisFunctions

export CreateMaxEntBasis, getBasisDerivative, EvaluateMaxEntBasis

# Maximum entropy basis functions
# ===============================

using  LineSearches, Optim, NLsolve, NearestNeighbors, LinearAlgebra

struct MaxEntBasis
	x::Matrix{Real};
	β::Real; # Constant β. Future version we can incorporate x_i dependent β
end


"""
CreateMaxEntBasis(x::Matrix; nn=3, γ=1)::MaxEntBasis

# DESCRIPTION

	Creates MaxEntBasis object with given points x, and tuning parameters nn, flag.
	This object is used to generate basis and derivatives.

# INPUT

	x::Matrix of size d x N, where d is the dimension of state space, N is the number of points
	nn is an integer to define number of nearest neighbours -- tune local vs global property
	flag tunes how distances from nn neighbours are used to define locality of maximum entropy basis functions
		flag = 0: minimum distance (default)
		flag = 1: maximum distance
		flag = 2: average distance
	a is scaling of distance based β, detault = 1

# OUTPUT

	MaxEntBasis	object with fields x::Matrix{Real}, and β::Vector{Real}, where β defines the locality of the gaussian RBs.	
"""
function CreateMaxEntBasis(x::Matrix;nn=3,γ=1)::MaxEntBasis
	# 1. We need to compute β, which is needed for computing the maxent basis functions.
	
	kdtree = KDTree(x)
	idxs, dists = knn(kdtree, x, nn, true); # Compute knn for all points in the data

	# # Compute β according to flag -- use list comprehension to do this efficiently
	# if flag == 0 # Minimum
	# 	b = [d[2] for d in dists];
	# elseif flag == 1 # Maximum 
	# 	b = [d[end] for d in dists];
	# elseif flag == 2 # Average
	# 	b = [sum(d[2:end])/(length(d)-1) for d in dists]; # Skip the first element, which is distance with self.
	# end
	# β = γ ./ b; # See paper 

	D = vcat(dists'...);
	D1 = vec(D[:,2:nn]);
	avgD = sum(D1)/length(D1); # Average nn distance. 
	β = γ/avgD; # See paper 
	
	return(MaxEntBasis(x,β));
end


function EvaluateMaxEntBasis(objϕ::MaxEntBasis, x::Array{T,1})::Array{T,2} where T <: Real
	xTilde = objϕ.x .- x;
	nBasis = size(xTilde,2);
	λ = getLambda(x,objϕ);
	ϕ = meBasis(λ,x,objϕ)'; # Return row vector
	return ϕ;
end

function priorFunction(x,β)
    f = exp(-β*norm(x)^2);
	return f;
end

function getLambda(x,objϕ)
	function costFunction(λ)
		phi = meBasis(λ,x,objϕ);
		nBasis = length(phi);
		d = length(x);

		J = zeros(d);
		xTilde = x .- objϕ.x;

		for i in 1:nBasis
			J += xTilde[:,i]*phi[i];
		end
		return J'*J
	end

	d = length(x)
	initial_x = zeros(d);
	sol = optimize(costFunction,initial_x,Newton());
	λ = Optim.minimizer(sol);
	# println("λ: $λ")
	return(λ);
end

# Generate maximum entropy-basis functions
# ========================================
function meBasis(λ, x, objϕ)
	xTilde = x .- objϕ.x;
	nBasis = size(xTilde,2);
	Z = zeros(nBasis);
	
	for i in 1:nBasis
		m = priorFunction(xTilde[:,i],objϕ.β);
		Z[i] = m*exp(-λ'*xTilde[:,i]);
	end
	phi = Z./sum(Z);
	return phi; 
end


# Compute gradient of maxent basis function
# =================================================================
# See following paper for technical details
#	Local maximum-entropy approximation schemes: 
#	A seamless bridge between finite elements and meshfree methods 
#   by M. Arroyo and M. Ortiz
# -----------------------------------------------------------------
function getBasisDerivative(objϕ,xEval,ϕ)
	xTilde = objϕ.x .- xEval;
	d, nBasis = size(xTilde);
	
	J = zeros(d,d);
	r = zeros(d);

	for i in 1:nBasis
		xTilde = (xEval-objϕ.x[:,i]);
		r .+= ϕ[i]*xTilde;
		J .+= ϕ[i]*xTilde*xTilde';
	end
	J = J - r*r';
	# invJ = inv(J); # replace
	invJ = J\I(d);
	∇ϕ = zeros(d,nBasis);
	for i in 1:nBasis
		∇ϕ[:,i] = -ϕ[i]*invJ*(xEval-objϕ.x[:,i]);
	end
	return ∇ϕ
end
end # module
