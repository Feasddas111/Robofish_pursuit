function [rotation_mat] = fishsim_op_euler2rot(phi, theta, psi)
    cos_phi = cos(phi);
    sin_phi = sin(phi);
    cos_theta = cos(theta);
    sin_theta = sin(theta);
    cos_psi = cos(psi);
    sin_psi = sin(psi);
    
    rotation_mat = zeros(3);
    rotation_mat(1,1) = cos_psi*cos_theta;
    rotation_mat(2,1) = sin_psi*cos_theta;
    rotation_mat(3,1) = -sin_theta;
    rotation_mat(1,2) = cos_psi*sin_theta*sin_phi - sin_psi*cos_phi;
    rotation_mat(2,2) = sin_psi*sin_theta*sin_phi + cos_psi*cos_phi;
    rotation_mat(3,2) = cos_theta*sin_phi;
    rotation_mat(1,3) = cos_psi*sin_theta*cos_phi + sin_psi*sin_phi;
    rotation_mat(2,3) = sin_psi*sin_theta*cos_phi - cos_psi*sin_phi;
    rotation_mat(3,3) = cos_theta*cos_phi;
    
end

