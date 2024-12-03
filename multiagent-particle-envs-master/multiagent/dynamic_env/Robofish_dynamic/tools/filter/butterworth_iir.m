clc;
clear all;
close all;
dt = 0.01; % ����Ƶ��: 100hz
f1 = 2;  % Ƶ�ʳɷ�1
f2 = 5;  % Ƶ�ʳɷ�2
f3 = 20; % Ƶ�ʳɷ�3
f4 = 40; % Ƶ�ʳɷ�4
X = [];
G = [];
for t = 0:dt:3
    g = -sin(f1*2*pi*t+0.1) + 0.2*sin(f2*2*pi*t+0.8);
    x = g - 0.3*sin(f3*2*pi*t+0.8) + 0.4*sin(f4*2*pi*t-0.8);
    G = [G;g];
    X = [X;x]; 
end
% ԭʼ����
subplot(4,1,1)
plot(G, 'g-', 'LineWidth', 2)
hold on
plot(X,'r-', 'LineWidth', 1)
hold on
legend('������', 'ԭʼ����')
% ��ֵ�˲�
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
legend('������', '��ֵ�˲�')
% IIR     
% b0*Y(k)-b1*Y(k-1)-b2*Y(k-2)-...=a0*X(k)+a1*X(k-1)+a2*X(k-2)+...
% Y(k)=(b1*Y(k-1)+b2*Y(k-2)+...+a0*X(k)+a1*X(k-1)+a2*X(k-2)+...)/b0
% һ�װ�����˹�˲���
[a,b]=butter(1,0.1,'low'); % 10hz/100hz=0.1
sX1 = butterworth_filter(X, a, b, 1);
subplot(4,1,3)
plot(G, 'g-', 'LineWidth', 2)
hold on
plot(sX1,'b-', 'LineWidth', 1)
hold on
legend('������', 'һ�װ�����˹')
% ���װ�����˹�˲���
[a,b]=butter(2,0.1,'low'); % 10hz/100hz=0.1
sX2 = butterworth_filter(X, a, b, 2);
subplot(4,1,4)
plot(G, 'g-', 'LineWidth', 2)
hold on
plot(sX2,'m-', 'LineWidth', 1)
hold on
legend('������', '���װ�����˹')
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