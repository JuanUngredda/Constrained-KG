%   ***************************************
%   *** Code written by German Gutierrez***
%   ***         gg92@cornell.edu        ***
%   ***   Edited by Jennifer Shih       ***
%   ***       jls493@cornell.edu        ***
%   ***************************************
%
%   updated July 23, 2014

%RETURNS:  Avg. and C.I. Halfwidth of total revenue and mean profit. 
% CURRENTLY set to 100000 repetitions, 3 periods each.

function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = RMITD(x, runlength, seed, true_val, ~)
% function [fn, FnVar, FnGrad, FnGradCov, constraint, ConstraintCov, ConstraintGrad, ConstraintGradCov] = RMITD(x, runlength, seed, other);
% x is a vector of reservations for each period, in order [b r2 r3 ... rT];
% runlength is the number of days of demand to simulate
% seed is the index of the substreams to use (integer >= 1)
% other is not usedR
%
%Note: RandStream.setGlobalStream(stream) can only be used for Matlab
%versions 2011 and later
%For earlier versions, use the method RandStream.setDefaultStream(stream)

FnGrad = NaN;
FnGradCov = NaN;
constraint = NaN;
ConstraintCov = NaN;
ConstraintGrad = NaN;
ConstraintGradCov = NaN;

b=x(1); 
x=[x 0]; 
x(1)=[]; % x(i) = r_(i+1), shifted for later ease of code; Note that r_(T+1) = 0

if (sum(x < 0) > 0)  || (runlength <= 0) || (runlength ~= round(runlength)) || (seed <= 0) || (round(seed) ~= seed),
    fprintf('x should be >= 0, runlength should be positive integer, seed must be a positive integer\n');
    fn = NaN;
    FnVar = NaN;

else
    
    % Provided data: 3 periods, prices and mean Demand as in problem statement.
    T = 3;
    price = [100 300 400];
    meanDemand = [50 20 30];
    cost = 80;
    
    k = 1;
    theta = 1/k;
    
    % Policy being used; c is the quantity purchased, x_t is the minimum number
    % of units that should be left for periods t+1, t+2, ...
    buy = b; 
    
    %Simulation 10000 times.
    nReps = runlength;
    
    revenue = zeros(nReps,1);
    
    % Generate a new stream for random numbers
    [XStream, YStream] = RandStream.create('mrg32k3a','NumStreams', 2);

    % Set the substream to the "seed"
    XStream.Substream = seed;
    YStream.Substream = seed;
    
    % Generate random X
    OldStream = RandStream.setGlobalStream(XStream);
    %OldStream = RandStream.setDefaultStream(XStream); %versions 2010 and earlier
    disp(true_val);
    if (true_val==1),
        X = normrnd(0 ,0, nReps);
        
    else
       
    	X = normrnd(0 ,1, nReps); %gamrnd(k, theta, 1, nReps);
    end
    % Generate random Y_j's
    RandStream.setGlobalStream(YStream); 
    %RandStream.setDefaultStream(YStream); %versions 2010 and earlier
    Y = exprnd(1, nReps, T);
    
    RandStream.setGlobalStream(OldStream); %Return to old stream
    %RandStream.setDefaultStream(OldStream); %versions 2010 and earlier
    
    parfor i = 1:nReps
        remainingCapacity = buy;
        for j = 1:T
            % Generate Demand
            D_t = meanDemand(j) + X(i)%*Y(i,j);
            % Accept Bookings
           
            sell = min(max(remainingCapacity-x(j),0),D_t);
            
            remainingCapacity = remainingCapacity - sell;
           
            revenue(i) = revenue(i) + price(j)*sell;
            
        end
    end

MeanRevenue = mean(revenue);
StdRev = std(revenue);
CIHalfWidth = norminv(0.975)*StdRev/sqrt(nReps);
fn = MeanRevenue - cost*buy;
FnVar = var(revenue)/nReps;

end
end

