using Revise, PyPlot, LinearAlgebra, Sobol, ForwardDiff, Random
include("../src/MaxEntropyBasisFunctions.jl"); # Our code.

nBasis = 10; nEval = 1000;

xData = 1.1*sort(reshape(2*rand(MersenneTwister(0), nBasis) .- 1,1,nBasis),dims=2);
# xData = reshape(1.05*collect(range(-1,1,length=nBasis)),1,nBasis);

xEval = reshape(collect(range(-1,1,length=nEval)),1,nEval);

d = size(xData,1);
# Generate mexent basis functions
Phi = zeros(nEval,nBasis);
derPhi = zeros(nEval,nBasis);
m = zeros(nEval,nBasis);

ϕ = CreateMaxEntBasis(xData,nn=5); 

@time for i ∈ 1:nEval
	Phi[i,:] = EvaluateMaxEntBasis(ϕ,xEval[:,i])
	derPhi[i,:] = getBasisDerivative(ϕ,xEval[:,i],Phi[i,:]);
	for j in 1:nBasis
		m[i,j] = priorFunction(xEval[:,i]-ϕ.x[:,j],ϕ.β);
	end
end

# Plot ϕ and ϕ_x
nCol = 5;
if  mod(nBasis,nCol) == 0
	nRow = div(nBasis,nCol);
else
	nRow = div(nBasis,nCol)+1;
end

# Find maximum
vals, inds = findmax(Phi,dims=1);

figure(1,figsize=(10,4)); clf(); 
pygui(true)
rc("text", usetex=true);
rc("font", serif="Helvetica");
rc("font", size=10);
for i in 1:nBasis
	subplot(nRow,nCol,i);
	# plot(xEval,Phi[:,i],linewidth=1.0); grid("on");
	plot(xEval',Phi[:,i],linewidth=1.0,color="b");
	plot(xEval',derPhi[:,i],linewidth=1.0,color="r"); 
	plot(xEval',m[:,i],linewidth=1.0,color="g"); 
	grid("on");

	# # Plots maximum point
	# X = xEval[1,inds[i][1]];
	# plot([X;X],[0;vals[i]],color="k",lw=0.5);
	
	local str = "\$" * "\\phi_" * "{$i}" * "\$";
	title(LaTeXString(str));
	# axis("off");
end
tight_layout();
legend((L"\phi",L"\phi_x",L"e^{-\beta x^2}"));
