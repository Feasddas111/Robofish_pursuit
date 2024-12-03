clc;
clear all;
close all;

%% 配置模型
FileName     = 'blogmodel_m';               % File describing the model structure.
Order           = [3 1 3];                        % Model orders [ny nu nx].
Parameters  = [1, -1]';       % Initial parameter vector.

InitialStates = zeros(3, 1);                % Initial states.
Ts            = 0;                                   % Time-continuous system.
nlgr = idnlgrey(FileName, Order, Parameters, InitialStates, Ts, ...
                        'Name', 'Blog Example Model',                            ...
                        'InputName',  {'Voltage'}',            ...
                        'InputUnit', {'V' }',                              ...
                        'OutputName', {'Position' 'Velocity' 'Acceleration'}',      ...
                        'OutputUnit', {'m' 'm/s' 'm/s^2'}',                          ...
                        'TimeUnit', 's');
            
nlgr = setinit(nlgr, 'Name', {'Position' 'Velocity' 'Acceleration'}');
nlgr = setinit(nlgr, 'Unit', {'m' 'm/s' 'm/s^2'}');

nlgr = setpar(nlgr, 'Name', {'Ka : coeff 1' 'Kb : coeff2' });
nlgr.Parameters(1).Fixed = false; 
nlgr.Parameters(1).Minimum = -10;
nlgr.Parameters(1).Maximum = 10;

present(nlgr)

%% 导入数据
dataset = load('blogexample.mat');
z = iddata(dataset.Output, dataset.Input, 0.01, 'Name', 'Actual Data');
z.InputName = {'Voltage'};
z.InputUnit = {'V' };
z.OutputName = {'Position' 'Velocity' 'Acceleration'};
z.OutputUnit = {'m' 'm/s' 'm/s^2'};
z.Tstart = 0;
z.TimeUnit = 's';

present(z);

%% 参数效果
X0init = zeros(3,1);
nlgr = setinit(nlgr, 'Value', num2cell(X0init));
figure(1)
compare(getexp(z, 1), nlgr, [], compareOptions('InitialCondition', X0init));

%% 估计参数
nlgr = nlgreyest(nlgr, getexp(z, 1), nlgreyestOptions('Display', 'on'));
figure(2)
compare(getexp(z, 1), nlgr, [], compareOptions('InitialCondition', X0init));
