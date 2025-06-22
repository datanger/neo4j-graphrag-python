% 跨脚本变量传递案例

global_var = 100;
result = use_global(global_var);

function out = use_global(val)
    out = val + 1;
end
