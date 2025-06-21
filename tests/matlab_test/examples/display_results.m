
            function display_results(varargin)
                % Function that displays results
                for i = 1:length(varargin)
                    fprintf('Result %d: %f
', i, varargin{i});
                end
            end
            