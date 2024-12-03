clc;
clear all;
close all;
load(fullfile(matlabroot, 'toolbox', 'ident', 'iddemos', 'data', 'dcmotordata'));
z = iddata(y, u, 0.1, 'Name', 'DC-motor');
z.InputName = 'Voltage';
z.InputUnit =  'V';
z.OutputName = {'Angular position', 'Angular velocity'};
z.OutputUnit = {'rad', 'rad/s'};
z.Tstart = 0;
z.TimeUnit = 's';
figure('Name', [z.Name ': Voltage input -> Angular position output']);
plot(z(:, 1, 1));   % Plot first input-output pair (Voltage -> Angular position).
figure('Name', [z.Name ': Voltage input -> Angular velocity output']);
plot(z(:, 2, 1));   % Plot second input-output pair (Voltage -> Angular velocity).


FileName      = 'dcmotor_m';       % File describing the model structure.
Order         = [2 1 2];           % Model orders [ny nu nx].
Parameters    = [1; 0.28];         % Initial parameters. Np = 2.
InitialStates = [0; 0];            % Initial initial states.
Ts            = 0;                 % Time-continuous system.
nlgr = idnlgrey(FileName, Order, Parameters, InitialStates, Ts, ...
                'Name', 'DC-motor');
set(nlgr, 'InputName', 'Voltage', 'InputUnit', 'V',               ...
          'OutputName', {'Angular position', 'Angular velocity'}, ...
          'OutputUnit', {'rad', 'rad/s'},                         ...
          'TimeUnit', 's');      
nlgr = setinit(nlgr, 'Name', {'Angular position' 'Angular velocity'});
nlgr = setinit(nlgr, 'Unit', {'rad' 'rad/s'});
nlgr = setpar(nlgr, 'Name', {'Time-constant' 'Static gain'});
nlgr = setpar(nlgr, 'Unit', {'s' 'rad/(V*s)'});      

nlgr.SimulationOptions.AbsTol = 1e-6;
nlgr.SimulationOptions.RelTol = 1e-5;

figure()
compare(z, nlgr);

nlgr = setinit(nlgr, 'Fixed', {false false}); % Estimate the initial states.
opt = nlgreyestOptions('Display', 'on');
nlgr = nlgreyest(z, nlgr, opt);

figure()
compare(z, nlgr);

figure()
pe(z, nlgr);