
            % Main script that demonstrates various relationship patterns
            
            % Call a function
            result1 = helper_function(5);
            
            % Call another script
            run('helper_script.m');
            
            % Define and use variables
            x = 10;
            y = x + 5;
            z = y * 2;
            
            % Call a function that modifies a variable
            [x, y] = modify_variables(x, y);
            
            % Use variables in a function call
            display_results(x, y, z);
            