function [theta, dtheta, ddtheta] = fishsim_tail_oscillation_model(t, freq, amp, bias, mode)
    persistent past_t;
    persistent cpg_x;
    persistent cpg_y;
    persistent cpg_k1;
    persistent cpg_k2;
    if isempty(past_t)
        past_t = 0;
        cpg_x = 0.0001;
        cpg_y = 0.0001;
        cpg_k1 = 50;
        cpg_k2 = 50;
    end
    freq = freq*2*pi; 
    if strcmp(mode, 'Sin')
        theta = bias + amp*sin(freq*t);
        dtheta = freq*amp*cos(freq*t);
        ddtheta = -freq*freq*amp*sin(freq*t);
    elseif strcmp(mode, 'CPG')
        dt = t ;
        x = cpg_x - bias;
        y = cpg_y;
        d_cpg_x = -freq*y + cpg_k1*x*(amp*amp - x*x - y*y);
        d_cpg_y = freq*x + cpg_k2*y*(amp*amp - x*x - y*y);
        cpg_x = cpg_x + d_cpg_x*dt;
        cpg_y = cpg_y + d_cpg_y*dt;
        dd_cpg_x = -freq*d_cpg_y + cpg_k1*d_cpg_x*(amp*amp - x*x - y*y) + cpg_k1*x*(-2*x*d_cpg_x-2*y*d_cpg_y);
        theta = cpg_x;
        dtheta = d_cpg_x;
        ddtheta = dd_cpg_x;
    end
end
