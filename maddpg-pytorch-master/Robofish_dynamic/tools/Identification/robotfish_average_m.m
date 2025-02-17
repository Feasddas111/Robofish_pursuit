function [dx, y] = robotfish_average_m(t, x, u, Mx, Mz, Ly, Kx, Kz, Ktau, Kt, KD, KL, rbp, mrbm, Kforce, varargin)
    g = 9.8;
    y = x;
    dx = [sin(x(2))*x(3)-cos(x(2))*x(4); ...
              x(5); ...
             -x(4)*x(5)+1/Mx*(-x(6)*g*sin(x(2))+abs(x(3))*x(3)*(-Kx-KD*cos(u(2))+KL*u(2)*sin(u(2)))+0.5*Kforce*Kt*u(1)*7/8); ...
             x(3)*x(5)+1/Mz*(x(6)*g*cos(x(2))-Kz*abs(x(4))*x(4)+abs(x(3))*x(3)*(-KD*sin(u(2))-KL*u(2)*cos(u(2)))); ...
             1/Ly*((Mx-Mz)*x(3)*x(4)-Ktau*abs(x(5))*x(5)+abs(x(3))*x(3)*(-KD*sin(u(2))-KL*u(2)*cos(u(2)))*rbp-mrbm*g*sin(x(2))); ...
             u(3)];
end

