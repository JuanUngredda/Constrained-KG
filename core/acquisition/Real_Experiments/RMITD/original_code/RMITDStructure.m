function [minmax d m VarNature VarBds FnGradAvail NumConstraintGradAvail, StartingSol, budget, ObjBd, OptimalSol] = RMITDStructure(NumStartingSol, seed)
%function [minmax d m VarNature VarBds FnGradAvail NumConstraintGradAvail, StartingSol] = RMITDStructure(NumStartingSol, seed);
% Inputs:
%	a) NumStartingSol: Number of starting solutions required. Integer, >= 0
%	b) seed: Seed for generating random starting solutions. Integer, >= 1
% Return structural information on optimization problem
%     a) minmax: -1 to minimize objective , +1 to maximize objective
%     b) d: positive integer giving the dimension d of the domain
%     c) m: nonnegative integer giving the number of constraints. All
%        constraints must be inequality constraints of the form LHS >= 0.
%        If problem is unconstrained (beyond variable bounds) then should be 0.
%     d) VarNature: a d-dimensional column vector indicating the nature of
%        each variable - real (0), integer (1), or categorical (2).
%     e) VarBds: A d-by-2 matrix, the ith row of which gives lower and
%        upper bounds on the ith variable, which can be -inf, +inf or any
%        real number for real or integer variables. Categorical variables
%        are assumed to take integer values including the lower and upper
%        bound endpoints. Thus, for 3 categories, the lower and upper
%        bounds could be 1,3 or 2, 4, etc.
%     f) FnGradAvail: Equals 1 if gradient of function values are
%        available, and 0 otherwise.
%     g) NumConstraintGradAvail: Gives the number of constraints for which
%        gradients of the LHS values are available. If positive, then those
%        constraints come first in the vector of constraints.
%     h) StartingSol: One starting solution in each row, or NaN if NumStartingSol=0.
%        Solutions generated as per problem writeup
%     i) budget: Column vector of suggested budgets, or NaN if none suggested
%     j) ObjBd is a bound (upper bound for maximization problems, lower
%        bound for minimization problems) on the optimal solution value, or
%        NaN if no such bound is known.
%     k) OptimalSol is a d dimensional column vector giving an optimal
%        solution if known, and it equals NaN if no optimal solution is known.

%   ***************************************
%   *** Written by Shane Henderson to   ***
%   *** use standard calling and random ***
%   *** number streams                  ***
%   *** Values updated by Victor Wu     ***
%   *** Edited by Jennifer Shih         ***
%   *** Edited by Bryan Chong           ***
%   ***************************************
% Last updated Sept 28, 2014 
T = 3; % Time Horizon
b = 100; % Quantity to buy (starting)
minmax = 1; % Maximize mean profit (-1 for minimize)
d = T; %T-1; % T-1 quantities to reserve for periods [t,T] for t=2..T
m = 0; % No constraints
VarNature = zeros(d, 1); % real variables
VarBds = ones(d,1)*[0, Inf]; % must reserve at most total amount bought
FnGradAvail = 0; % No Derivative is computed
NumConstraintGradAvail = 0; % No constraints
budget = [200; 1000; 3000; 10000]; 
ObjBd=NaN;
OptimalSol=NaN;

if (NumStartingSol < 0) || (NumStartingSol ~= round(NumStartingSol)) || (seed <= 0) || (round(seed) ~= seed),
    fprintf('NumStartingSol should be integer >= 0, seed must be a positive integer\n');
    StartingSol = NaN;
else
    if (NumStartingSol == 0),
        StartingSol = NaN;
    elseif (NumStartingSol == 1),
        StartingSol = [100, 50, 30]; %b, r2, r3
    else
        OurStream = RandStream.create('mlfg6331_64'); % Use a different generator from simulation code to avoid stream clashes
        OurStream.Substream = seed;
        OldStream = RandStream.setGlobalStream(OurStream);
        StartingSol = [ones(NumStartingSol,1)*100, unifrnd(40,60,NumStartingSol,1), unifrnd(20,40,NumStartingSol,1)]; % b=100; Continuous Uniform pairs, first component is ~U(40, 60); second is ~U(20, 40)
        RandStream.setGlobalStream(OldStream); % Restore previous stream
    end %if NumStartingSol
end %if inputs ok