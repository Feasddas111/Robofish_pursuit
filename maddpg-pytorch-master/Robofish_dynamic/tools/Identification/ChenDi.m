 
%% 1. Represent Exp_Data as an iddata Object
tic;
 %ID_Data:an iddata Objec;
 %Input_u:��������
 %Speed��ʵ����������
 Filename_Control = 'Control_Data_w=14.2689_PhaseDiff_70.xlsx';
 Filename_Exp = 'Speed_Data_Ks=100_w=28_PhaDiff=70_N.xlsx';
 [ID_Data,Input_u,Speed,zi] = Generation_Exp_iddata(Filename_Control,Filename_Exp); %���ý�ʵ�����ݱ�Ϊ an iddata Object����

%% 2. Represent the Robot_Arm dynamics using an idnlgrey object.
% finding proper initial parameter values for the robot arm requires some additional effort
% [dx,y] = Lag_Eq(t,x,u,Ks,mac,Iac,cf0,cf12,cf3,cd0,cd12,cd3,varargin) %%
% ע�����һ�varargin ��������

FileName      = 'Lag_Eq';               % File describing the model structure.
Order         = [2 6 8];                    % Model orders [ny nu nx]. ����u��
Parameters    = [100*(5.615e-4) 1.2 1.2 ...%Ks, mac,Iac     
                  0.05 0.2 0.2 ...% cf0,cf12,cf3
                  0.5 0.5 5 ...% cd0,cd12,cd3
                  ]';       % Initial parameter vector�����ĳ�ʼֵ.
%InitialStates = zeros(8, 1);                % ״̬������Initial states.
InitialStates = [0;0;0;0;0;0;0;0];     % ״̬������Initial states.
Ts            = 0;                          % Time-continuous system.

nlgr_Robotfish = idnlgrey(FileName, Order, Parameters, InitialStates, Ts, ...
                'Name', 'Robot Fish',                            ... % ʵ�����ݵ�����
                'InputName', {'P1' 'dP1' 'ddP1' 'P2' 'dP2' 'ddP2'},            ...
                'InputUnit', {'rad' 'rad/s' 'rad/s^2' 'rad' 'rad/s' 'rad/s^2'},                              ...
                'OutputName', {'Swimming Speed Vx' 'Swimming Speed Vy'},      ...
                'OutputUnit', {'cm/s' 'cm/s'},                          ...
                'TimeUnit', 's');
%   
% Specify names and units of the initial states. ״̬����������
nlgr_Robotfish = setinit(nlgr_Robotfish, 'Name', {'Position of J0 in X' ...
                              'Position of J0 in Y' ...
                              'Angular position difference between X and Link 0' ...                                      ...
                              'Angular position difference between X and Link 4' ...                                  ...
                              'Linear velocity of J0 in X'...
                              'Linear velocity of J0 in Y'...
                              'Angular velocity of J0'...
                              'Angular velocity of J3'...
                              }');
nlgr_Robotfish = setinit(nlgr_Robotfish, 'Unit', {'cm' 'cm' 'rad' 'rad' 'cm/s' 'cm/s' 'rad/s' 'rad/s'});

% Specify names and units of the parameters. ���������ƺ���
nlgr_Robotfish = setpar(nlgr_Robotfish, 'Name', {'Ks: Spring constant(stiffness)'        ... % 1.
                      'mac: Added mass coefficient'                  ... % 2.
                      'Iac: Added moment of inertia coefficient'     ... % 3.
                      'cf0: X drag coefficient 0'                    ... % 4.
                      'cf12: X drag coefficient 1,2'                    ... % 5.
                      'cf3: X drag coefficient 3'                    ... % 6.
                      'cd0: Y drag coefficient 0'                    ... % 7.
                      'cd12: Y drag coefficient 1,2'                    ... %8.
                      'cd3: Y drag coefficient 3'                    ... % 9.
                     });

% ģ�Ͳ����ı�ʶ����
   nlgr_Robotfish.Parameters(1).Fixed = true;
for parno1 = 2:9   % Fix the first six parameters.  ǰ������������Ϊ�̶��ģ�����Ҫ��ʶ�Ĳ���
    nlgr_Robotfish.Parameters(parno1).Fixed = false; % ��2����10 ��������Ҫ��ʶ
end 
   nlgr_Robotfish.Parameters(2).Minimum = 1.01; % ��2����3����ʶ���������� Ӧ��<=�����Ĵ���С
    nlgr_Robotfish.Parameters(2).Maximum = 4; % ��2����3����ʶ���������� Ӧ��>=�����ĳ�ֵ 
   % Fix the first six parameters.  ǰ������������Ϊ�̶��ģ�����Ҫ��ʶ�Ĳ���
    nlgr_Robotfish.Parameters(3).Minimum = 1.01; % ��2����3����ʶ���������� Ӧ��<=�����Ĵ���С
    nlgr_Robotfish.Parameters(3).Maximum = 4; % ��2����3����ʶ���������� Ӧ��>=�����ĳ�ֵ 
for parno4 = 4:6   % Fix the first six parameters.  ǰ������������Ϊ�̶��ģ�����Ҫ��ʶ�Ĳ���
    nlgr_Robotfish.Parameters(parno4).Minimum = 0.01; % ��4����11����ʶ����������   Ӧ��<=�����Ĵ���С
    nlgr_Robotfish.Parameters(parno4).Maximum = 1; % ��4����11����ʶ����������  Ӧ��>=�����ĳ�ֵ
end 

for parno5 = 7:8   % Fix the first six parameters.  ǰ������������Ϊ�̶��ģ�����Ҫ��ʶ�Ĳ���
    nlgr_Robotfish.Parameters(parno5).Minimum = 0.1; % ��4����11����ʶ����������   Ӧ��<=�����Ĵ���С
    nlgr_Robotfish.Parameters(parno5).Maximum = 10; % ��4����11����ʶ����������  Ӧ��>=�����ĳ�ֵ
end 

    nlgr_Robotfish.Parameters(9).Minimum = 0.1; % ��2����3����ʶ���������� Ӧ��<=�����Ĵ���С
    nlgr_Robotfish.Parameters(9).Maximum = 10; % ��2����3����ʶ���������� Ӧ��>=�����ĳ�ֵ 

 present(nlgr_Robotfish)           % �鿴������Ϣ

%% Parameter Estimation ��ʼ������ʶ
% �������
% nlgr.SimulationOptions.AbsTol = 1e-6;
% nlgr.SimulationOptions.RelTol = 1e-5;
opt = nlgreyestOptions;
opt.Display = 'on';
%opt.SearchOptions.MaxIterations = 150;
nlgr_Robotfish = nlgreyest(ID_Data,nlgr_Robotfish, opt);

present(nlgr_Robotfish);
toc