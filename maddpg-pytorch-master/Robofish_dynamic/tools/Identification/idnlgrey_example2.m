clc;
clear all;
close all;
FileName      = 'robotarm_c';               % File describing the model structure.
Order         = [1 1 5];                    % Model orders [ny nu nx].
Parameters    = [ 0.00986346744839  0.74302635727901 ...
                  3.98628540790595  3.24015074090438 ...
                  0.79943497008153  0.03291699877416 ...
                  0.17910964111956  0.61206166914114 ...
                 20.59269827430799  0.00000000000000 ...
                  0.06241814047290 20.23072060978318 ...
                  0.00987527995798]';       % Initial parameter vector.
InitialStates = zeros(5, 1);                % Initial states.
Ts            = 0;                          % Time-continuous system.
nlgr = idnlgrey(FileName, Order, Parameters, InitialStates, Ts, ...
                'Name', 'Robot arm',                            ...
                'InputName', 'Applied motor torque',            ...
                'InputUnit', 'Nm',                              ...
                'OutputName', 'Angular velocity of motor',      ...
                'OutputUnit', 'rad/s',                          ...
                'TimeUnit', 's');
nlgr = setinit(nlgr, 'Name', {'Angular position difference between the motor and the gear-box' ...
                       'Angular position difference between the gear-box and the arm'   ...
                       'Angular velocity of motor'                                      ...
                       'Angular velocity of gear-box'                                   ...
                       'Angular velocity of robot arm'}');
nlgr = setinit(nlgr, 'Unit', {'rad' 'rad' 'rad/s' 'rad/s' 'rad/s'});
nlgr = setpar(nlgr, 'Name', {'Fv   : Viscous friction coefficient'            ... % 1.
                      'Fc   : Coulomb friction coefficient'            ... % 2.
                      'Fcs  : Striebeck friction coefficient'          ... % 3.
                      'alpha: Striebeck smoothness coefficient'        ... % 4.
                      'beta : Friction smoothness coefficient'         ... % 5.
                      'J    : Total moment of inertia'                 ... % 6.
                      'a_m  : Motor moment of inertia scale factor'    ... % 7.
                      'a_g  : Gear-box moment of inertia scale factor' ... % 8.
                      'k_g1 : Gear-box stiffness parameter 1'          ... % 9.
                      'k_g3 : Gear-box stiffness parameter 3'          ... % 10.
                      'd_g  : Gear-box damping parameter'              ... % 11.
                      'k_a  : Arm structure stiffness parameter'       ... % 12.
                      'd_a  : Arm structure damping parameter'         ... % 13.
                     });
nlgr = setpar(nlgr, 'Minimum', num2cell(zeros(size(nlgr, 'np'), 1)));   % All parameters >= 0!
for parno = 1:6   % Fix the first six parameters.
    nlgr.Parameters(parno).Fixed = true;
end

load(fullfile(matlabroot, 'toolbox', 'ident', 'iddemos', 'data', 'robotarmdata'));
z = iddata({ye yv1 yv2 yv3}, {ue uv1 uv2 uv3}, 0.5e-3, 'Name', 'Robot arm');
z.InputName = 'Applied motor torque';
z.InputUnit = 'Nm';
z.OutputName = 'Angular velocity of motor';
z.OutputUnit = 'rad/s';
z.ExperimentName = {'Estimation' 'Validation 1'  'Validation 2' 'Validation 3'};
z.Tstart = 0;
z.TimeUnit = 's';
present(z);

figure('Name', [z.Name ': input-output data'],...
   'DefaultAxesTitleFontSizeMultiplier',1,...
   'DefaultAxesTitleFontWeight','normal',...
   'Position',[100 100 900 600]);
for i = 1:z.Ne
    zi = getexp(z, i);
    subplot(z.Ne, 2, 2*i-1);   % Input.
    plot(zi.u);
    title([z.ExperimentName{i} ': ' zi.InputName{1}],'FontWeight','normal');
    if (i < z.Ne)
        xlabel('');
    else
        xlabel([z.Domain ' (' zi.TimeUnit ')']);
    end
    subplot(z.Ne, 2, 2*i);     % Output.
    plot(zi.y);
    title([z.ExperimentName{i} ': ' zi.OutputName{1}],'FontWeight','normal');
    if (i < z.Ne)
        xlabel('');
    else
        xlabel([z.Domain ' (' zi.TimeUnit ')']);
    end
end

zred = z(1:round(zi.N/10));
nlgr = setinit(nlgr, 'Fixed', {true true false false false});
X0 = nlgr.InitialStates;
[X0.Value] = deal(zeros(1, 4), zeros(1, 4), [ye(1) yv1(1) yv2(1) yv3(1)], ...
    [ye(1) yv1(1) yv2(1) yv3(1)], [ye(1) yv1(1) yv2(1) yv3(1)]);
[~, X0init] = predict(zred, nlgr, [], X0);
num2cell(X0init(:, 1))
nlgr = setinit(nlgr, 'Value', num2cell(X0init(:, 1)));
figure()
compare(z, nlgr, [], compareOptions('InitialCondition', X0init));
% 
% nlgr = nlgreyest(nlgr, getexp(z, 1), nlgreyestOptions('Display', 'on'));
% 
% X0init(:, 1) = cell2mat(getinit(nlgr, 'Value'));
% X0 = nlgr.InitialStates;
% [X0.Value] = deal(zeros(1, 3), zeros(1, 3), [yv1(1) yv2(1) yv3(1)], ...
%     [yv1(1) yv2(1) yv3(1)], [yv1(1) yv2(1) yv3(1)]);
% [yp, X0init(:, 2:4)] = predict(getexp(zred, 2:4), nlgr, [], X0);
% figure()
% compare(z, nlgr, [], compareOptions('InitialCondition', X0init));
% figure;
% pe(z, nlgr, peOptions('InitialCondition',X0init));