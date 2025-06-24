function [out1, out2] = modify_variables(in1, in2)
    % Function that modifies input variables
    out1 = in1 * 2;
    out2 = in2 + 10;

    % Internal variable with same name as inputs
    x = out1 + out2;
    y = x / 2;
end
