clc;
clear all;
close all;

%% 配置模型
FileName     = 'robotfish_average_m';               % File describing the model structure.

Order           = [6 3 6];                        % Model orders [ny nu nx].
%Parameters  = [3.0241, 3.0241, 0.0486, 2.0, 10.0, 1.0, 0.00628, 1.5, 1.5, -0.1, 0.02]';       % Initial parameter vector.
Parameters  = [3.0241, 3.0241, 0.0486, 2.0, 10.0, 1.0, 0.00319, 1.74, 2.14, -0.15, 0.04, 1]';       % Initial parameter vector.
Parameters  = [3.0241, 3.0241, 0.0486, 2.0, 10.0, 1.0, 0.00628, 0.37, 0.46, -0.15, 0.023, 1]';       % Initial parameter vector.
Parameters  = [3.0241, 3.0241, 0.0486, 2.0, 10.0, 1.0, 0.00369, 0.839, 1.502, -0.15, 0.0298, 175.858]';       % Initial parameter vector.
Parameters  = [3.0241, 3.0241, 0.0486, 2.0, 10.0, 1.0, 0.00369, 0.839, 1.502, -0.15, 0.0298, 117]';       % Initial parameter vector.

InitialStates = zeros(6, 1);                % Initial states.
Ts            = 0;                                   % Time-continuous system.
nlgr = idnlgrey(FileName, Order, Parameters, InitialStates, Ts, ...
                        'Name', 'Robot Fish Depth Model',                            ...
                        'InputName',  {'Tail beat frequency' 'Pectoral fin deflection angle' 'Pumping speed'}',            ...
                        'InputUnit', {'Hz' 'rad' 'kg/s' }',                              ...
                        'OutputName', {'Depth' 'Pitch angle' 'X-axis velocity' 'Z-axis velocity' 'Picth angular velocity' 'Mass of ballast system'}',      ...
                        'OutputUnit', {'m' 'rad' 'm/s' 'm/s' 'rad/s' 'kg'}',                          ...
                        'TimeUnit', 's');
            
nlgr = setinit(nlgr, 'Name', {'Depth' 'Pitch angle' 'X-axis velocity' 'Z-axis velocity' 'Picth angular velocity' 'Mass of ballast system'}');
nlgr = setinit(nlgr, 'Unit', {'m' 'rad' 'm/s' 'm/s' 'rad/s' 'kg'}');
nlgr = setpar(nlgr, 'Name', {'Mx    : X-axis mass'            ... % 1.
                                             'Mz    : Z-axis mass'            ... % 2.
                                             'Ly     : Y-axis inertial moment'          ... % 3.
                                             'Kx     : X-axis drag coefficient'        ... % 4.
                                             'Kz     : Z-axis drag coefficient'         ... % 5.
                                             'Ktau : Y-axis damping torque coefficient'                 ... % 6.
                                             'Kt     : Tail force coefficient'    ... % 7.
                                             'KD    : Pectoral fin darg force coefficient' ... % 8.
                                             'KL     : Pectoral fin lift force coefficient'          ... % 9.
                                             'rbp   : Distance between COG and pectoral fin'          ... % 10.
                                             'mrbm   : Mass plus the distance between COG and COB'          ... % 11.
                                             'Kforce : Scale factor of tail force'
                     });
% nlgr = setpar(nlgr, 'Minimum', num2cell(zeros(size(nlgr, 'np'), 1)));   % All parameters >= 0!
for parno = 1:11   % Fix the first six parameters.
    nlgr.Parameters(parno).Fixed = true;
end
% nlgr.Parameters(4).Fixed = false;
% nlgr.Parameters(4).Minimum = 0;
% nlgr.Parameters(7).Fixed = true;
% nlgr.Parameters(8).Fixed = true;
% nlgr.Parameters(10).Minimum = -0.15;
% nlgr.Parameters(10).Maximum = 0;
present(nlgr)

%% 导入数据
dataset1 = load('tail-pectsine-average.mat');
dataset2 = load('tail-pectsine-average.mat');
dataset3 = load('tail-pectsine-average.mat');

z = iddata({dataset1.ID_X_DATA dataset2.ID_X_DATA dataset3.ID_X_DATA}, ...
                  {dataset1.ID_U_DATA dataset2.ID_U_DATA dataset3.ID_U_DATA}, 0.01, 'Name', 'Robot Fish');
z.InputName = {'Tail beat frequency' 'Pectoral fin deflection angle' 'Pumping speed'};
z.InputUnit = {'Hz' 'rad' 'kg/s' };
z.OutputName = {'Depth' 'Pitch angle' 'X-axis velocity' 'Z-axis velocity' 'Picth angular velocity' 'Mass of ballast system'};
z.OutputUnit = {'m' 'rad' 'm/s' 'm/s' 'rad/s' 'kg'};
z.ExperimentName = {'TailSine' 'PectSine'  'Tail-PectSine'};
z.Tstart = 0;
z.TimeUnit = 's';
present(z);

%% 参数效果
X0init = kron(ones(1,3),[-0.5; 0.0; 0.0; 0.0; 0.0; 0.0]);
nlgr = setinit(nlgr, 'Value', num2cell(X0init(:, 1)));
figure(1)
compare(getexp(z, 3), nlgr, [], compareOptions('InitialCondition', X0init));

%% 估计参数
nlgr = nlgreyest(nlgr, getexp(z, 3), nlgreyestOptions('Display', 'on'));
figure(2)
compare(getexp(z, 3), nlgr, [], compareOptions('InitialCondition', X0init));
