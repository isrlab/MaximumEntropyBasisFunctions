using Revise, PyPlot, LinearAlgebra, Sobol, ForwardDiff
using MaximumEntropyBasisFunctions
using ISRLutils

# # With random points
# lb = [-1 -1]; ub = [1 1];
# p = ISRLutils.cornerPoints(lb,ub)
# s = SobolSeq(lb,ub);
# skip(s,100); 
# xData = [p' hcat([next!(s) for i = 1:30]...)];
# xData = hcat([next!(s) for i = 1:20]...);

# With uniform grid
αMax = 3; nData = [3,3]; nEval = 50*[1,1]; 
lb = [-1,-1]*αMax; ub = [1,1]*αMax;
xData = ISRLutils.GenerateNDGrid(lb,ub,nData);
xEval = ISRLutils.GenerateNDGrid(0.99*lb,0.99*ub,nEval);

# Crete me basis object
ϕ = CreateMaxEntBasis(xData,nn=4); 

# Evaluate mexent basis functions at xEval
nBasis = size(xData,2);
nEval1 = size(xEval,2)
Phi = zeros(nEval1,nBasis);
derPhi = zeros(2*nEval1,nBasis);
@time for i ∈ 1:nEval1
	#  Phi1[i,:] = getBasis(xData,xEval[:,i]);
	 Phi[i,:] = EvaluateMaxEntBasis(ϕ,xEval[:,i])
	 derPhi[(2*i-1):2*i,:] = getBasisDerivative(ϕ,xEval[:,i],Phi[i,:]);
end

## Plotting
nCol = 5;
nRow = div(nBasis,nCol)+1;

rc("text", usetex=true);
rc("font", serif="Helvetica");
rc("font", size=9);

figure(1); clf(); pygui(true);
figure(2); clf(); pygui(true);
figure(3); clf(); pygui(true);

XX = reshape(xEval[1,:],nEval[1],nEval[2]);
YY = reshape(xEval[2,:],nEval[1],nEval[2]);
cmin = minimum(derPhi);
cmax = maximum(derPhi);

for i in 1:nBasis
    figure(1); subplot(nRow,nCol,i);
	pcolor(XX,YY,reshape(Phi[:,i], nEval[1],nEval[2]),shading="auto",ec="face");
	# surf(XX,YY,reshape(Phi[:,i], nEval[1],nEval[2]),cmap=PyPlot.cm.coolwarm);
	scatter(xData[1,:],xData[2,:],s=3,fc="r",ec="none");
	str = "\$" * "\\phi_" * "{$i}" * "\$";
	title(LaTeXString(str));
	axis("equal"); axis("off");  #grid("on");

	# ϕ_x1
	figure(2); subplot(nRow,nCol,i);
	pcolor(XX,YY,reshape(derPhi[1:2:end,i], nEval[1],nEval[2]),shading="auto",ec="face", vmin=cmin, vmax=cmax);
	# surf(XX,YY,reshape(derPhi[1:2:end,i], nEval[1],nEval[2]));
	scatter(xData[1,:],xData[2,:],s=3,fc="r",ec="none");
	str = "\$" * "\\frac{\\partial \\phi_" * "{$i}" * "}{\\partial x_1}\$";
	title(LaTeXString(str));
	axis("equal"); axis("off");  #grid("on");

	# ϕ_x2
	figure(3); subplot(nRow,nCol,i);
	pcolor(XX,YY,reshape(derPhi[2:2:end,i], nEval[1],nEval[2]),shading="auto",ec="face",vmin=cmin, vmax=cmax);
	# surf(XX,YY,reshape(derPhi[2:2:end,i], nEval[1],nEval[2]));
	scatter(xData[1,:],xData[2,:],s=3,fc="r",ec="none");
	# str = "\$" * "\\phi_" * "{$i}" * "\$";
	str = "\$" * "\\frac{\\partial \\phi_" * "{$i}" * "}{\\partial x_2}\$";
	title(LaTeXString(str));
	axis("equal"); axis("off");  #grid("on");
end

figure(1);tight_layout(); #savefig("maxent2D.pdf",dpi=300);
figure(2);tight_layout(); #savefig("maxent2Ddx1.pdf",dpi=300);
figure(3);tight_layout(); #savefig("maxent2Ddx2.pdf",dpi=300);

println("Plotting done ...");
# savefig("maxent2D.pdf",dpi=300);
