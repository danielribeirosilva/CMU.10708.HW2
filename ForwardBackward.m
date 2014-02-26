%HW2 Daniel Ribeiro Silva
%drsilva
% Forward Backward Algorithm

function ForwardBackward()

%load data
load('hmm_params.mat');

%constants from problem
Q = 3;   %total hidden states
T = 100; %total used timesteps

%predicted values
forward_predictions = zeros(Q,T);
backward_predictions = zeros(Q,T);

%get forward values

%first timestep (t=1) - P(Z_1|X_1) \propto P(X_1|Z_1)P(Z_1) = emission*prior
observed_output = price_change(1);
emission_prob_given_observed_output = emission(:,observed_output);
forward_predictions(:,1) = prior.*emission_prob_given_observed_output;

%for remaining timesteps (t>1) = P(Z_t|X_t,Z_t-1) \propto P(X_t|Z_t)P(Z_t|Z_t-1) = emission*transition
for t=2:T
    observed_output = price_change(t);
    emission_prob_given_observed_output = emission(:,observed_output);
    previous_states = forward_predictions(:,t-1);
    transition_prob_given_prev_states = transition'*previous_states;
    forward_predictions(:,t) = emission_prob_given_observed_output.*transition_prob_given_prev_states;
end

forward_prob = dot(forward_predictions(:,T),ones(Q,1));
%disp(forward_prob);
%disp(forward_predictions);

%get backward values

%first timestep (t=T) - P(Z_1|X_1) \propto P(X_1|Z_1)P(Z_1) = emission*prior
backward_predictions(:,T) = ones(Q,1);

%for remaining timesteps (t>1) = P(Z_t|X_t,Z_t-1) \propto P(X_t|Z_t)P(Z_t|Z_t-1) = emission*transition
for t=(T-1):-1:1
    observed_output = price_change(t+1);
    emission_prob_given_observed_output = emission(:,observed_output);
    previous_states = backward_predictions(:,t+1);
    backward_predictions(:,t) = transition*(emission_prob_given_observed_output.*previous_states);
end

%backward_prob = dot(forward_predictions(:,1).*prior,emission(:,price_change(1)));
%disp(backward_prob);

finalPred = forward_predictions.*backward_predictions/forward_prob;

plot(finalPred');
legend('bull','bear','stable');

end

