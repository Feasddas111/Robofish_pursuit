function [dx, y] = blogmodel_m(t, x, u, Ka, Kb, varargin)
  
  y = [x(1); x(2); x(3)];
  
  dx = [Ka*sin(x(2)); ...
            Kb*x(3)^2; ...
            u];
        
end