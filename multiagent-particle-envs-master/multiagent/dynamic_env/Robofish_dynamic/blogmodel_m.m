function [position, velocity, omege] = blogmodel_m(p_pos, p_vel, p_omega, u)
    head_infos = YAML.read('headparam_dynamicattach.yml');
    TAIL_CPG_AMP = 15;
    period = 0.2;

    velocity = [p_vel(1);p_vel(2);0];
    head_infos.rigidhead.velocity = velocity;

    angvel = [0;0;p_omega(1)];
    head_infos.rigidhead.angvel = angvel;

    tail_amp = TAIL_CPG_AMP/180*pi;

    m_b = head_infos.rigidhead.mass*eye(3);
    m_ad =  head_infos.rigidhead.mass_ad;
    m = m_b + m_ad;

    I_b = head_infos.rigidhead.inertia_tensor;
    I_ad =  head_infos.rigidhead.inertia_tensor_ad;
    I = I_b + I_ad;

    r_bm = reshape(head_infos.rigidhead.r_bm, 3,1);
    r_bt = reshape(head_infos.rigidhead.r_bt, 3,1);

    r_bm_crossmat = fishsim_op_crossmat(r_bm);
    H_body = [ I, m_b*r_bm_crossmat;
                      -m_b*r_bm_crossmat, m];

    rho = head_infos.physicsparam.rho;
    g = head_infos.physicsparam.g;

    Kv = reshape( head_infos.rigidhead.Kv,3,1);
    Kw = reshape( head_infos.rigidhead.Kw,3,1);

    velocity = [p_vel(1);p_vel(2);0];
    angvel = [0;0;p_omega(1)];

    F_tail_old = zeros(3,1);
    M_tail_old = zeros(3,1);

    [theta_f, dtheta_f, ddtheta_f] = fishsim_tail_oscillation_model(period, u(1), tail_amp, u(2), 'CPG');

    F_hydro = Kv.*abs(velocity).*velocity;
    M_hydro = Kw.*abs(angvel).*angvel;

    cflen = head_infos.caudalfin.length;
    cfhei = head_infos.caudalfin.height;
    cfvm = 0.25*pi*rho*cfhei^2;
    F_tail = -0.5*cfvm*cflen^2*ddtheta_f*[sin(theta_f);-cos(theta_f);0];
    M_tail = [0; 0; F_tail(2)*r_bt(1)-1/3*cfvm*cflen^3*ddtheta_f];

    tau = 0.02;
    b = 1;
    max_force = 10;
    if(abs(F_tail(1)) < max_force &&  abs(F_tail(2)) < max_force)
        d_F_tail = -1/tau*F_tail_old + b/tau*F_tail;
        F_tail_old = F_tail_old + d_F_tail*period;
        d_M_tail = -1/tau*M_tail_old + b/tau*M_tail;
        M_tail_old = M_tail_old + d_M_tail*period;
    end
    F_tail = F_tail_old;
    M_tail = M_tail_old;

    F_ext = F_hydro  + F_tail;
    M_ext = M_hydro  + M_tail;

    Hmat = H_body;
    Hinv = inv(Hmat);
    avmat = fishsim_op_crossmat(angvel);
    r_bm_crossmat = fishsim_op_crossmat(r_bm);
    K_body = [I*avmat, m_b*r_bm_crossmat*avmat;
                 -m_b*avmat*r_bm_crossmat, m*avmat];
    Kmat = K_body;

    U = [angvel; velocity];
    u_1 = [M_ext; F_ext];
    U_dot = -Hinv*Kmat *U + Hinv*u_1;

    acceleration = U_dot(4:6);
    angaccel = U_dot(1:3);

    velocity = velocity + acceleration*period;
    angvel = angvel + angaccel*period;
    omege = angvel(3);

    P = [p_pos(1);p_pos(2);0] ;
    position = P + velocity*period;

end
