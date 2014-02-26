%HW2 Daniel Ribeiro Silva
%drsilva
%Forward Algorithm


function pred_Y = forwardAlgorithm(Q,T,price_change,prior,emission,transition)

%predicted values
forward_predictions = zeros(Q,T);

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

pred_Y = forward_predictions(:,T)/sum(forward_predictions(:,T));

end
