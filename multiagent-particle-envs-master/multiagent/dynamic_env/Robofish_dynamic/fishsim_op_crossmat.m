function [crossmat] = fishsim_op_crossmat(x)
    crossmat = zeros(3,3);
    crossmat(1,2) = -x(3);
    crossmat(1,3) = x(2);
    crossmat(2,1) = x(3);
    crossmat(2,3) = -x(1);
    crossmat(3,1) = -x(2);
    crossmat(3,2) = x(1);
end

