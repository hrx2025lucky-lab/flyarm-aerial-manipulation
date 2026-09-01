function modelPath = build_flyarm_control_block_diagram()
%BUILD_FLYARM_CONTROL_BLOCK_DIAGRAM Build a paper-style control diagram.
%
% This model is for explanation and thesis figures. The runnable numerical
% verification remains flyarm_closed_loop_compare.slx, which calls the MATLAB
% closed-loop step functions directly.

model = "flyarm_control_block_diagram";
root = fileparts(fileparts(mfilename("fullpath")));
modelPath = fullfile(root, "simulink", model + ".slx");

if bdIsLoaded(model)
    set_param(model, "Dirty", "off");
    close_system(model, 0);
end
if exist(modelPath, "file")
    delete(modelPath);
end

new_system(model);
open_system(model);
set_param(model, "Solver", "FixedStepDiscrete", "FixedStep", "0.01", "StopTime", "10.0");

stepPath = "simulink/Sources/Step";
subsystemPath = "simulink/Ports & Subsystems/Subsystem";
scopePath = "simulink/Sinks/Scope";

add_block(stepPath, model + "/Reference", ...
    "Time", "0.5", "Before", "0.2", "After", "1.0", ...
    "Position", [60 170 120 210]);

add_stage(subsystemPath, model + "/Controller", [190 150 320 230], "lightBlue");
add_stage(subsystemPath, model + "/Motor Mixer", [390 150 520 230], "green");
add_stage(subsystemPath, model + "/Motor Dynamics", [590 150 740 230], "cyan");
add_stage(subsystemPath, model + "/6DOF Plant", [810 150 940 230], "yellow");
add_block(scopePath, model + "/Scope", "Position", [1010 165 1060 215]);

add_line(model, "Reference/1", "Controller/1", "autorouting", "on");
add_line(model, "Controller/1", "Motor Mixer/1", "autorouting", "on");
add_line(model, "Motor Mixer/1", "Motor Dynamics/1", "autorouting", "on");
add_line(model, "Motor Dynamics/1", "6DOF Plant/1", "autorouting", "on");
add_line(model, "6DOF Plant/1", "Scope/1", "autorouting", "on");

note = ["Paper-style control-loop diagram for the flyarm MATLAB/Simulink layer.", ...
    "Reference -> Controller -> Motor Mixer -> Motor Dynamics -> 6DOF Plant -> Scope.", ...
    "Use flyarm_closed_loop_compare.slx for executable PID/LADRC numerical validation."];
Simulink.Annotation(model, strjoin(note, newline));

save_system(model, modelPath);
close_system(model, 0);
fprintf("Saved paper-style Simulink control diagram: %s\n", modelPath);
end

function add_stage(sourcePath, blockPath, position, color)
add_block(sourcePath, blockPath, "Position", position, "BackgroundColor", color);
end
