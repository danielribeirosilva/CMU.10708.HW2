%HW2 Daniel Ribeiro Silva
%drsilva
%Viterbi Algorithm

function Viterbi()

load('hmm_params.mat');

%constants from problem
Q = 3;   %total hidden states
T = 100; %total used timesteps

%predicted values
V = zeros(Q,T);
path = zeros(Q,T);

%case t=1
observed_output = price_change(1);
V(:,1) = prior.*emission(:,observed_output);
path(:,1) = (1:Q)';

%case t>1
for t=2:T
   observed_output = price_change(t);
   [max_val,idx] = max( bsxfun(@times,V(:,t-1),transition') );
   max_val = max_val';
   max_val = max_val .* emission(:,observed_output);
   V(:,t) = max_val;
   path = path(idx,:);
   path(:,t) = (1:Q)';
end

%get best path overall
[max_val,idx] = max(V(:,T));
best_path = path(idx,:);
plot(best_path);
axis([1 T 0.5 3.5])

%matlab solution (doesn't consider prior)
%STATES = hmmviterbi(price_change,transition,emission);
%disp(STATES);

end