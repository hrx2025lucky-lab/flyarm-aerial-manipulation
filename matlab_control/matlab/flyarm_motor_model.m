function omegaNext = flyarm_motor_model(omega, omegaCmd, dt, P)
%FLYARM_MOTOR_MODEL First-order motor/rotor speed response with limits.

omega = omega(:);
omegaCmd = omegaCmd(:);
omegaCmd = min(max(omegaCmd, 0), P.motor.maxOmega);

alpha = 1 - exp(-dt / P.motor.timeConstant);
rawNext = omega + alpha * (omegaCmd - omega);

maxStep = P.motor.maxOmegaDot * dt;
step = min(max(rawNext - omega, -maxStep), maxStep);
omegaNext = omega + step;
omegaNext = min(max(omegaNext, 0), P.motor.maxOmega);
end
