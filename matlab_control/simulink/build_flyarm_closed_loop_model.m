function modelPath = build_flyarm_closed_loop_model()
%BUILD_FLYARM_CLOSED_LOOP_MODEL Build runnable PID/LADRC closed-loop model.

model = "flyarm_closed_loop_compare";
root = fileparts(fileparts(mfilename("fullpath")));
addpath(fullfile(root, "matlab"));

if bdIsLoaded(model)
    close_system(model, 0);
end
if exist(fullfile(root, "simulink", model + ".slx"), "file")
    delete(fullfile(root, "simulink", model + ".slx"));
end

new_system(model);
open_system(model);
set_param(model, "Solver", "FixedStepDiscrete", "FixedStep", "0.01", "StopTime", "10.0");

clockPath = "simulink/Sources/Clock";
toWorkspacePath = "simulink/Sinks/To Workspace";
interpPath = interpreted_block_path();

add_block(clockPath, model + "/Clock", "Position", [60 140 90 170]);

add_block(interpPath, model + "/PID closed loop", ...
    "MATLABFcn", "flyarm_simulink_step_pid", ...
    "OutputDimensions", "25", ...
    "SampleTime", "0.01", ...
    "Position", [170 75 330 125]);
add_block(interpPath, model + "/LADRC closed loop", ...
    "MATLABFcn", "flyarm_simulink_step_ladrc", ...
    "OutputDimensions", "25", ...
    "SampleTime", "0.01", ...
    "Position", [170 185 330 235]);

add_block(toWorkspacePath, model + "/pid_out", ...
    "VariableName", "pid_out", "SaveFormat", "Array", ...
    "Position", [430 82 520 118]);
add_block(toWorkspacePath, model + "/ladrc_out", ...
    "VariableName", "ladrc_out", "SaveFormat", "Array", ...
    "Position", [430 192 520 228]);

add_line(model, "Clock/1", "PID closed loop/1");
add_line(model, "Clock/1", "LADRC closed loop/1");
add_line(model, "PID closed loop/1", "pid_out/1");
add_line(model, "LADRC closed loop/1", "ladrc_out/1");

note = ["Runnable closed-loop comparison", ...
    "Clock drives two interpreted MATLAB function blocks.", ...
    "Each block executes: reference -> PID/LADRC -> mixer -> motor lag -> 6DOF dynamics.", ...
    "Outputs are 25x1 vectors logged to workspace as pid_out and ladrc_out."];
Simulink.Annotation(model, strjoin(note, newline));

modelPath = fullfile(root, "simulink", model + ".slx");
save_system(model, modelPath);
close_system(model, 0);
fprintf("Saved runnable Simulink closed-loop model: %s\n", modelPath);
end

function path = interpreted_block_path()
load_system("simulink");
blocks = find_system("simulink/User-Defined Functions", "SearchDepth", 1);
idx = find(contains(blocks, "Interpreted") & contains(blocks, "MATLAB"), 1);
if isempty(idx)
    error("build_flyarm_closed_loop_model:MissingBlock", "Cannot find Interpreted MATLAB Function block.");
end
path = blocks{idx};
end
