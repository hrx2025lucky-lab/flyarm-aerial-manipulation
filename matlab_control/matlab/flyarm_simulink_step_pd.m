function y = flyarm_simulink_step_pd(t)
%FLYARM_SIMULINK_STEP_PD Legacy wrapper; now routes to PID baseline.

persistent simState
[y, simState] = flyarm_simulink_step_impl(t, "pid", simState);
end
