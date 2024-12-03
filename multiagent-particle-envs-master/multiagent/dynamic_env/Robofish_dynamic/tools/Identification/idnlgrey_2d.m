clc;
clear all;
close all;

%% ����ģ��
FileName     = 'robotfish2d_m';               % File describing the model structure.

Order           = [6 2 6];                        % Model orders [ny nu nx].
Parameters  = [3.0241, 3.0241, 0.0187, 2.0, 8.0, 1.0, 0.00558583, 0.111366, -0.193981]';       % Initial parameter vector.


InitialStates = zeros(6, 1);                % Initial states.
Ts            = 0;                                   % Time-continuous system.
nlgr = idnlgrey(FileName, Order, Parameters, InitialStates, Ts, ...
                        'Name', 'Robot Fish Depth Model',                            ...
                        'InputName',  {'Tail joint joint' 'Tail joint angular acceleration'}',            ...
                        'InputUnit', {'rad' 'rad/s^2'}',                              ...
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
                                             'L    : Tail fin length' ... % 8.
                                             'rbt   : Distance between COG and tail fin'          ... % 10.
                     });
% nlgr = setpar(nlgr, 'Minimum', num2cell(zeros(size(nlgr, 'np'), 1)));   % All parameters >= 0!
for parno = 1:6  % Fix the first six parameters.
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

%% ��������
dataset1 = load('tail-2dsine.mat');
dataset2 = load('tail-2dsine.mat');
dataset3 = load('tail-2dsine.mat');

z = iddata({dataset1.ID_X_DATA dataset2.ID_X_DATA dataset3.ID_X_DATA}, ...
                  {dataset1.ID_U_DATA1 dataset2.ID_U_DATA1 dataset3.ID_U_DATA1}, 0.01, 'Name', 'Robot Fish');
z.InputName = {'Tail joint joint' 'Tail joint angular acceleration'};
z.InputUnit = {'rad' 'rad/s^2'};
z.OutputName = {'Xpos' 'Ypos' 'Yaw' 'X-axis velocity' 'Y-axis velocity' 'Yaw angular velocity'};
z.OutputUnit = {'m' 'm' 'rad' 'm/s' 'm/s' 'rad/s'};
z.ExperimentName = {'TailSine' 'PectSine'  'Tail-PectSine'};
z.Tstart = 0;
z.TimeUnit = 's';
present(z);

%% ����Ч��
X0init = kron(ones(1,3),[0.0; 0.0; 0.0; 0.0; 0.0; 0.0]);
nlgr = setinit(nlgr, 'Value', num2cell(X0init(:, 1)));
figure(1)
compare(getexp(z, 1), nlgr, [], compareOptions('InitialCondition', X0init));

%% ���Ʋ���
nlgr = nlgreyest(nlgr, getexp(z, 1), nlgreyestOptions('Display', 'on'));
figure(2)
compare(getexp(z, 1), nlgr, [], compareOptions('InitialCondition', X0init));
present(nlgr)