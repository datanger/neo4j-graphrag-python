function display_results(varargin)
    % Function that displays multiple results
    for i = 1:length(varargin)
        fprintf('Result %d: %f\n', i, varargin{i});
    end
end
