function [dx, y] = robotfish2d_m(t, x, u, Mx, My, Lz, Kx, Ky, Ktauz, Kt, L, rbt, varargin)
    g = 9.8;
    y = x;
    dx = [cos(x(3))*x(4)-sin(x(3))*x(5);
              sin(x(3))*x(4)+cos(x(3))*x(5);
              x(6);
              x(5)*x(6)+(-Kx)/Mx*abs(x(4))*x(4)-(Kt)/Mx*u(2)*sin(u(1));
              -x(4)*x(6)-(Ky)/My*abs(x(5))*x(5)+Kt/My*u(2)*cos(u(1));
              -(Mx-My)/Lz*x(4)*x(5)-Ktauz/Lz*abs(x(6))*x(6)-Kt/Lz*(-rbt*u(2)*cos(u(1))+2/3*L*u(2));];
end

