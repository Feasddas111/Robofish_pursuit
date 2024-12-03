function [dx, y] = robotfish2d_average_m(t, x, u, Mx, My, Lz, Kx, Ky, Ktauz, Kt, rbt, Kforce, Kmoment, varargin)
    g = 9.8;
    y = x;
    alpha_A = 0.7854;
    u1 = alpha_A^2*u(1)^2*(1-0.5*u(2)^2-1/8*alpha_A^2);
    u2 = alpha_A^2*u(1)^2*u(2);
    dx = [cos(x(3))*x(4)-sin(x(3))*x(5);
              sin(x(3))*x(4)+cos(x(3))*x(5);
              x(6);
              x(5)*x(6)+(-Kx)/Mx*abs(x(4))*x(4)+(Kt*Kforce)/Mx/2*u1;
              -x(4)*x(6)-(Ky)/My*abs(x(5))*x(5)+(Kt*Kforce)/My/2*u2;
              -(Mx-My)/Lz*x(4)*x(5)-Ktauz/Lz*abs(x(6))*x(6)-Kt*Kmoment*rbt/Lz/2*u2];
end

