function y = flyarm_simulink_step_ladrc(t)
%FLYARM_SIMULINK_STEP_LADRC Interpreted MATLAB Function wrapper for Simulink.

persistent simState
[y, simState] = flyarm_simulink_step_impl(t, "ladrc", simState);
end
