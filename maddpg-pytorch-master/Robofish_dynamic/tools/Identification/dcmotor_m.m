function [dx, y] = dcmotor_m(t, x, u, tau, k, varargin)
  % Output equations.
  y = [x(1);                         ... % Angular position.
       x(2)                          ... % Angular velocity.
      ];
  % State equations.
  dx = [x(2);                        ... % Angular velocity.
        -(1/tau)*x(2)+(k/tau)*u(1)   ... % Angular acceleration.
       ];
end