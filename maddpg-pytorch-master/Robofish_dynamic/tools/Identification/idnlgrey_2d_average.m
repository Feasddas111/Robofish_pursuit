clc;
clear all;
close all;

%% 配置模型
FileName     = 'robotfish2d_average_m';               % File describing the model structure.

Order           = [6 2 6];                        % Model orders [ny nu nx].
Parameters  = [3.0241, 3.0241, 0.0187, 2.0, 8.0, 1.0, 0.00558583, -0.193981, 39.0436, -5.18858]';       % Initial parameter vector.


InitialStates = zeros(6, 1);                % Initial states.
Ts            = 0;                                   % Time-continuous system.
nlgr = idnlgrey(FileName, Order, Parameters, InitialStates, Ts, ...
                        'Name', 'Robot Fish Depth Model',                            ...
                        'InputName',  {'Tail beat frequency' 'Tail beat bias'}',            ...
                        'InputUnit', {'Hz' 'rad'}',                              ...
                        'OutputName', {'Xpos' 'Ypos' 'Yaw' 'X-axis velocity' 'Y-axis velocity' 'Yaw angular velocity'}',      ...
                        'OutputUnit', {'m' 'm' 'rad' 'm/s' 'm/s' 'rad/s'}',                          ...
                        'TimeUnit', 's');
 
nlgr = setinit(nlgr, 'Name', {'Xpos' 'Ypos' 'Yaw' 'X-axis velocity' 'Y-axis velocity' 'Yaw angular velocity'}');
nlgr = setinit(nlgr, 'Unit', {'m' 'm' 'rad' 'm/s' 'm/s' 'rad/s'}');
nlgr = setpar(nlgr, 'Name', {'Mx    : X-axis mass'            ... % 1.
                                             'My    : Y-axis mass'            ... % 2.
                                             'Lz     : Z-axis inertial moment'          ... % 3.
                                             'Kx     : X-axis drag coefficient'        ... % 4.
                                             'Ky     : Y-axis drag coefficient'         ... % 5.
                                             'Ktauz : Z-axis damping torque coefficient'                 ... % 6.
                                             'Kt     : Tail force coefficient'    ... % 7.
                                             'rbt   : Distance between COG and tail fin'          ... % 10.
                                             'Kforce : Scale factor of tail force' ...
                                             'Kmoment : Scale factor of tail moment' 
                     });
% nlgr = setpar(nlgr, 'Minimum', num2cell(zeros(size(nlgr, 'np'), 1)));   % All parameters >= 0!
for parno = 1:8  % Fix the first six parameters.
    nlgr.Parameters(parno).Fixed = true;
end
% nlgr.Parameters(8).Fixed = true;
% nlgr.Parameters(9).Fixed = true;
% nlgr.Parameters(4).Fixed = false;
% nlgr.Parameters(4).Minimum = 0;
% nlgr.Parameters(7).Fixed = true;
% nlgr.Parameters(8).Fixed = true;
% nlgr.Parameters(10).Minimum = -0.15;
% nlgr.Parameters(10).Maximum = 0;
present(nlgr)

%% 导入数据
dataset1 = load('tail-2d-average.mat');
dataset2 = load('tail-2d-average.mat');
dataset3 = load('tail-2d-average.mat');

z = iddata({dataset1.ID_X_DATA dataset2.ID_X_DATA dataset3.ID_X_DATA}, ...
                  {dataset1.ID_U_DATA2 dataset2.ID_U_DATA2 dataset3.ID_U_DATA2}, 0.01, 'Name', 'Robot Fish');
z.InputName = {'Tail beat frequency' 'Tail beat bias'};
z.InputUnit = {'Hz' 'rad'};
z.OutputName = {'Xpos' 'Ypos' 'Yaw' 'X-axis velocity' 'Y-axis velocity' 'Yaw angular velocity'};
z.OutputUnit = {'m' 'm' 'rad' 'm/s' 'm/s' 'rad/s'};
z.ExperimentName = {'TailSine' 'PectSine'  'Tail-PectSine'};
z.Tstart = 0;
z.TimeUnit = 's';
present(z);

%% 参数效果
X0init = kron(ones(1,3),[0.0; 0.0; 0.0; 0.0; 0.0; 0.0]);
nlgr = setinit(nlgr, 'Value', num2cell(X0init(:, 1)));
figure(1)
compare(getexp(z, 1), nlgr, [], compareOptions('InitialCondition', X0init));

%% 估计参数
nlgr = nlgreyest(nlgr, getexp(z, 1), nlgreyestOptions('Display', 'on'));
figure(2)
compare(getexp(z, 1), nlgr, [], compareOptions('InitialCondition', X0init));
present(nlgr)