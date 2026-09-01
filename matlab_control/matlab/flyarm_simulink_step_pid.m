function y = flyarm_simulink_step_pid(t)
%FLYARM_SIMULINK_STEP_PID Interpreted MATLAB Function wrapper for Simulink.

persistent simState
[y, simState] = flyarm_simulink_step_impl(t, "pid", simState);
end
