clc;
clear all;
close all;

period = 0.01;
T = 10;
Ka = 1.21;
Kb = -0.6;

x1 = 0;
x2 = 0;
x3 = 0;

Input = [];
Output = [];

for t = 0:period:T
    
    u = sin(2*pi/10*t);
    
    x1_p = x1 + Ka*sin(x2)*period;
    x2_p = x2 + Kb*x3^2*period;
    x3_p = x3 + u*period;
    x1 = x1_p;
    x2 = x2_p;
    x3 = x3_p;
    y1 = x1;
    y2 = x2;
    y3 = x3;
    
    Input = [Input; u];
    Output = [Output; y1,y2,y3];
end

save('blogexample.mat', 'Input', 'Output');
