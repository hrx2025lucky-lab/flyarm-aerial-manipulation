function state = flyarm_pid_init(~, dt, gains)
%FLYARM_PID_INIT Initialize flight PID integral states.

if nargin < 2 || isempty(dt)
    dt = 0.01;
end
if nargin < 3 || isempty(gains)
    gains = default_pid_gains();
end

state = struct();
state.dt = dt;
state.gains = gains;
state.intPos = zeros(3, 1);
state.intAtt = zeros(3, 1);
end

function gains = default_pid_gains()
gains = struct();
gains.KpPos = [1.0; 1.0; 2.0];
gains.KiPos = [0.02; 0.02; 0.40];
gains.KdPos = [1.2; 1.2; 2.2];
gains.KpAtt = [5.0; 5.0; 2.5];
gains.KiAtt = [0.6; 0.6; 0.08];
gains.KdAtt = [1.6; 1.6; 0.8];
gains.maxAcc = [3.0; 3.0; 4.0];
gains.maxIntPos = [0.8; 0.8; 0.8];
gains.maxIntAtt = [0.6; 0.6; 0.45];
end
