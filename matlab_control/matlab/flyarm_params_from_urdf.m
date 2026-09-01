function P = flyarm_params_from_urdf(urdfPath)
%FLYARM_PARAMS_FROM_URDF Extract first-version flyarm parameters from URDF.
%
% P = flyarm_params_from_urdf(urdfPath) reads the project paper-facing
% parallel gripper URDF and returns the physical parameters needed by the
% MATLAB control model. The full articulated inertia is intentionally not
% solved here; the first-version flight controller uses the LADRC-aligned
% control inertia recorded in the Isaac project while preserving URDF mass
% and rotor geometry.

arguments
    urdfPath (1, :) char
end

doc = xmlread(urdfPath);
P = struct();
P.source.urdfPath = urdfPath;
P.gravity = 9.81;

links = parse_links(doc);
joints = parse_joints(doc);

P.links = links;
P.joints = joints;
P.mass.total = sum([links.mass]);
P.mass.byLink = links;

baseIdx = find(strcmp({links.name}, "base_link"), 1);
if isempty(baseIdx)
    error("flyarm_params_from_urdf:MissingBase", "URDF does not contain base_link.");
end
P.inertia.J_base_urdf = links(baseIdx).inertia;

% Isaac LADRC notes use b0 ~= 1/J: roll/pitch 80, yaw 130. This is the
% control inertia used by the first-version Simulink/MATLAB controller.
P.inertia.J_control = diag([1 / 80, 1 / 80, 1 / 130]);

rotorOrder = ["fl", "fr", "rr", "rl"];
rotorNames = strings(4, 1);
rotorPos = zeros(4, 3);
for i = 1:numel(rotorOrder)
    key = rotorOrder(i) + "_rotor_joint";
    idx = find(strcmp({joints.name}, key), 1);
    if isempty(idx)
        error("flyarm_params_from_urdf:MissingRotor", "Missing rotor joint %s.", key);
    end
    rotorNames(i) = key;
    rotorPos(i, :) = joints(idx).origin_xyz;
end
P.rotors.names = rotorNames;
P.rotors.position_b = rotorPos;
% Conventional alternating directions for an X quadrotor in [fl fr rr rl].
P.rotors.spin_dir = [1; -1; 1; -1];

armNames = ["base_rotate_joint", "axis_B_joint", "axis_C_joint", "axis_D_joint"];
P.arm.joints = select_joints(joints, armNames);

gripperIdx = find(contains({joints.name}, "gripper") & ~contains({joints.name}, "mount"));
P.gripper.joints = joints(gripperIdx);

P.motor.thrustToWeight = 1.9;
P.motor.maxOmega = 900.0;       % rad/s, first-version estimate
P.motor.timeConstant = 0.05;    % s, small rotor/ESC first-order lag estimate
P.motor.maxOmegaDot = 2.0e4;    % rad/s^2
maxThrustTotal = P.motor.thrustToWeight * P.mass.total * P.gravity;
P.motor.kf = (maxThrustTotal / 4) / (P.motor.maxOmega ^ 2);
P.motor.hoverOmega = sqrt((P.mass.total * P.gravity / 4) / P.motor.kf);
P.motor.km = 0.015 * P.motor.kf;

P.control.maxTilt = 0.45;       % rad
P.control.maxThrust = maxThrustTotal;
P.control.maxMoment = [0.40; 0.40; 0.25];
end

function links = parse_links(doc)
nodes = doc.getElementsByTagName("link");
links = repmat(struct("name", '', "mass", 0, "inertial_xyz", zeros(1, 3), ...
    "inertia", zeros(3, 3)), 1, nodes.getLength());

for i = 0:nodes.getLength()-1
    node = nodes.item(i);
    links(i+1).name = char(node.getAttribute("name"));
    inertial = first_child(node, "inertial");
    if isempty(inertial)
        continue;
    end
    massNode = first_child(inertial, "mass");
    originNode = first_child(inertial, "origin");
    inertiaNode = first_child(inertial, "inertia");
    if ~isempty(massNode)
        links(i+1).mass = str2double(char(massNode.getAttribute("value")));
    end
    if ~isempty(originNode) && originNode.hasAttribute("xyz")
        links(i+1).inertial_xyz = parse_vec(char(originNode.getAttribute("xyz")));
    end
    if ~isempty(inertiaNode)
        ixx = attr_double(inertiaNode, "ixx");
        ixy = attr_double(inertiaNode, "ixy");
        ixz = attr_double(inertiaNode, "ixz");
        iyy = attr_double(inertiaNode, "iyy");
        iyz = attr_double(inertiaNode, "iyz");
        izz = attr_double(inertiaNode, "izz");
        links(i+1).inertia = [ixx, ixy, ixz; ixy, iyy, iyz; ixz, iyz, izz];
    end
end
end

function joints = parse_joints(doc)
nodes = doc.getElementsByTagName("joint");
joints = repmat(struct("name", '', "type", '', "parent", '', "child", '', ...
    "origin_xyz", zeros(1, 3), "origin_rpy", zeros(1, 3), "axis", zeros(1, 3), ...
    "lower", NaN, "upper", NaN, "effort", NaN, "velocity", NaN), 1, nodes.getLength());

for i = 0:nodes.getLength()-1
    node = nodes.item(i);
    joints(i+1).name = char(node.getAttribute("name"));
    joints(i+1).type = char(node.getAttribute("type"));
    parent = first_child(node, "parent");
    child = first_child(node, "child");
    origin = first_child(node, "origin");
    axis = first_child(node, "axis");
    limit = first_child(node, "limit");
    if ~isempty(parent)
        joints(i+1).parent = char(parent.getAttribute("link"));
    end
    if ~isempty(child)
        joints(i+1).child = char(child.getAttribute("link"));
    end
    if ~isempty(origin)
        if origin.hasAttribute("xyz")
            joints(i+1).origin_xyz = parse_vec(char(origin.getAttribute("xyz")));
        end
        if origin.hasAttribute("rpy")
            joints(i+1).origin_rpy = parse_vec(char(origin.getAttribute("rpy")));
        end
    end
    if ~isempty(axis) && axis.hasAttribute("xyz")
        joints(i+1).axis = parse_vec(char(axis.getAttribute("xyz")));
    end
    if ~isempty(limit)
        joints(i+1).lower = attr_double(limit, "lower");
        joints(i+1).upper = attr_double(limit, "upper");
        joints(i+1).effort = attr_double(limit, "effort");
        joints(i+1).velocity = attr_double(limit, "velocity");
    end
end
end

function out = select_joints(joints, names)
out = repmat(joints(1), 1, numel(names));
for i = 1:numel(names)
    idx = find(strcmp({joints.name}, names(i)), 1);
    if isempty(idx)
        error("flyarm_params_from_urdf:MissingJoint", "Missing joint %s.", names(i));
    end
    out(i) = joints(idx);
end
end

function child = first_child(node, tag)
list = node.getElementsByTagName(tag);
if list.getLength() == 0
    child = [];
else
    child = list.item(0);
end
end

function v = parse_vec(txt)
v = sscanf(txt, "%f").';
if numel(v) ~= 3
    error("flyarm_params_from_urdf:BadVector", "Expected 3-vector, got '%s'.", txt);
end
end

function value = attr_double(node, name)
if node.hasAttribute(name)
    value = str2double(char(node.getAttribute(name)));
else
    value = NaN;
end
end
