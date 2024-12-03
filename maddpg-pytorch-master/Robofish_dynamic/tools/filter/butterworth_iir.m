clc;
clear all;
close all;
dt = 0.01; % 采样频率: 100hz
f1 = 2;  % 频率成分1
f2 = 5;  % 频率成分2
f3 = 20; % 频率成分3
f4 = 40; % 频率成分4
X = [];
G = [];
for t = 0:dt:3
    g = -sin(f1*2*pi*t+0.1) + 0.2*sin(f2*2*pi*t+0.8);
    x = g - 0.3*sin(f3*2*pi*t+0.8) + 0.4*sin(f4*2*pi*t-0.8);
    G = [G;g];
    X = [X;x]; 
end
% 原始数据
subplot(4,1,1)
plot(G, 'g-', 'LineWidth', 2)
hold on
plot(X,'r-', 'LineWidth', 1)
hold on
legend('无噪声', '原始数据')
% 均值滤波
sX0 = X;
alpha = 0.1;
for ii=2:length(X)
    sX0(ii) = (1-alpha)*sX0(ii-1) + alpha*X(ii);
end
subplot(4,1,2)
plot(G, 'g-', 'LineWidth', 2)
hold on
plot(sX0,'c-', 'LineWidth', 1)
hold on
legend('无噪声', '均值滤波')
% IIR     
% b0*Y(k)-b1*Y(k-1)-b2*Y(k-2)-...=a0*X(k)+a1*X(k-1)+a2*X(k-2)+...
% Y(k)=(b1*Y(k-1)+b2*Y(k-2)+...+a0*X(k)+a1*X(k-1)+a2*X(k-2)+...)/b0
% 一阶巴特沃斯滤波器
[a,b]=butter(1,0.1,'low'); % 10hz/100hz=0.1
sX1 = butterworth_filter(X, a, b, 1);
subplot(4,1,3)
plot(G, 'g-', 'LineWidth', 2)
hold on
plot(sX1,'b-', 'LineWidth', 1)
hold on
legend('无噪声', '一阶巴特沃斯')
% 二阶巴特沃斯滤波器
[a,b]=butter(2,0.1,'low'); % 10hz/100hz=0.1
sX2 = butterworth_filter(X, a, b, 2);
subplot(4,1,4)
plot(G, 'g-', 'LineWidth', 2)
hold on
plot(sX2,'m-', 'LineWidth', 1)
hold on
legend('无噪声', '二阶巴特沃斯')
function smooth_data = butterworth_filter(data, a, b, order)
    smooth_data = data;
    for ii = order+1:length(data)
        parta = 0;
        for jj = 1:order+1
        	parta = parta + a(jj)*data(ii-jj+1);
        end
        partb = 0;
        for kk = 2:order+1
        	partb = partb - b(kk)*smooth_data(ii-kk+1);
        end
        smooth_data(ii) = parta + partb;
    end
end