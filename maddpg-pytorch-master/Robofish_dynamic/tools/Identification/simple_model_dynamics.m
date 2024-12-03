clc;
clear all;
close all;

T = 50;
period = 0.01;

dataset = load('tail-pectsine-new.mat');
U = dataset.ID_U_DATA;
Xexample = dataset.ID_X_DATA;
Parameters  = [3.0241, 3.0241, 0.0486, 2.0, 10.0, 1.0, 0.00369, 0.839, 1.502, -0.15, 0.0298];  
Mx = Parameters(1);
Mz = Parameters(2);
Ly = Parameters(3);
Kx = Parameters(4);
Kz = Parameters(5);
Ktau = Parameters(6);
Kt = Parameters(7);
KD = Parameters(8);
KL = Parameters(9);
rbp = Parameters(10);
mrbm = Parameters(11);
g = 9.8;

h = -0.5;
theta = 0;
u = 0;
w = 0;
q = 0;
mba = 0;

X = [];
Time = [];
cnt = 0;
for t = 0:period:T
    cnt = cnt + 1;
    u1 = U(cnt,1);
    u2 = U(cnt,2);
    u3 = U(cnt,3);
    u4 = U(cnt,4);
    
    h_1 = h+(sin(theta)*u-cos(theta)*w)*period;
    theta_1 = theta + q*period;
    u_1 = u + (-w*q+1/Mx*(-mba*g*sin(theta)+abs(u)*u*(-Kx-KD*cos(u3)+KL*u3*sin(u3))-Kt*u2*sin(u1)))*period;
    w_1 = w + (u*q+1/Mz*(mba*g*cos(theta)-Kz*abs(w)*w+abs(u)*u*(-KD*sin(u3)-KL*u3*cos(u3))))*period;
    %q_1 = q + (1/Ly*((Mx-Mz)*u*w-Ktau*abs(q)*q+abs(u)*u*(-KD*sin(u3)-KL*u3*cos(u3))*rbp))*period;
    q_1 = q + (1/Ly*((Mx-Mz)*u*w-Ktau*abs(q)*q+abs(u)*u*(-KD*sin(u3)-KL*u3*cos(u3))*rbp-mrbm*g*sin(theta)))*period;
    mba_1 = mba + u4*period;
    
    h = h_1;
    theta = theta_1;
    u = u_1;
    w = w_1;
    q = q_1;
    mba = mba_1;
    
    X = [X; h, theta, u, w, q, mba];
    Time = [Time; t];
end

figure(1)
subplot(6,1,1)
plot(Time, X(:,1),'b');
hold on
plot(Time, Xexample(:,1),'--r');
grid on
ylabel('h');

subplot(6,1,2)
plot(Time, X(:,2),'b');
hold on
plot(Time, Xexample(:,2),'--r');
grid on
ylabel('theta');

subplot(6,1,3)
plot(Time, X(:,3),'b');
hold on
plot(Time, Xexample(:,3),'--r');
grid on
ylabel('u');

subplot(6,1,4)
plot(Time, X(:,4),'b');
hold on
plot(Time, Xexample(:,4),'--r');
grid on
ylabel('w');

subplot(6,1,5)
plot(Time, X(:,5),'b');
hold on
plot(Time, Xexample(:,5),'--r');
grid on
ylabel('q');

subplot(6,1,6)
plot(Time, X(:,6),'b');
hold on
plot(Time, Xexample(:,6),'--r');
grid on
ylabel('mba');