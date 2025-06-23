from neo4j_graphrag.experimental.components.code_extractor.matlab.schema import GraphSchema, NodeType, PropertyType, RelationshipType

# Schema definitions for MATLAB code analysis
SCHEMA = GraphSchema(
    node_types=[
        NodeType(
            label="Function",
            description="A code function definition",
            properties=[
                PropertyType(name="name", type="STRING", description="Name of the function"),
                PropertyType(name="file_path", type="STRING", description="Path to the file containing the function"),
                PropertyType(name="line_range", type="STRING", description="Line range where the function is defined"),
                PropertyType(name="description", type="STRING", description="Function description from docstring"),
                PropertyType(name="parameters", type="STRING", description="List of function parameters"),
                PropertyType(name="returns", type="STRING", description="List of return values"),
            ],
        ),
        NodeType(
            label="Variable",
            description="A variable used in the code",
            properties=[
                PropertyType(name="name", type="STRING", description="Name of the variable"),
                PropertyType(name="file_path", type="STRING", description="Path to the file where the variable is defined"),
                PropertyType(name="scope_id", type="STRING", description="ID of the scope (script or function) where this variable is defined"),
                PropertyType(name="scope_type", type="STRING", description="Type of scope: 'script' or 'function'"),
                PropertyType(name="line_range", type="LIST", description="List of tuples containing variable usage in script and corresponding line range, each tuple element is like (context, start_line-end_line)"),
            ],
        ),
        NodeType(
            label="Script",
            description="A code script file",
            properties=[
                PropertyType(name="name", type="STRING", description="Name of the script"),
                PropertyType(name="file_path", type="STRING", description="Path to the script file"),
                PropertyType(name="description", type="STRING", description="Script description"),
            ],
        ),
    ],
    relationship_types=[
        RelationshipType(
            label="CALLS",
            description="A function or script calls another function or script"
        ),
        RelationshipType(
            label="USES",
            description="A function or script uses a variable which is defined in another function or script",
        ),
        RelationshipType(
            label="DEFINES",
            description="A script defines a variable or a function and a function defines a variable",
        ),
        RelationshipType(
            label="ASSIGNED_TO",
            description="A variable is assigned to another variable",
        ),
        RelationshipType(
            label="MODIFIES",
            description="A function or script modifies a variable that was defined in another scope",
        ),
    ],
    patterns=[
        ("Function", "CALLS", "Function"),
        ("Function", "CALLS", "Script"),
        ("Script", "CALLS", "Function"),
        ("Script", "CALLS", "Script"),
        ("Function", "USES", "Variable"),
        ("Script", "USES", "Variable"),
        ("Function", "DEFINES", "Variable"),
        ("Script", "DEFINES", "Variable"),
        ("Script", "DEFINES", "Function"),
        ("Function", "MODIFIES", "Variable"),
        ("Script", "MODIFIES", "Variable"),
        ("Variable", "ASSIGNED_TO", "Variable"),
    ],
)


REQUIREMENTS = """
**关系模式说明：**

1. **CALLS 关系**：
   - `Script -> Function`: 脚本调用函数 (如 main_script.m 调用 helper_function)
   - `Script -> Script`: 脚本调用其他脚本 (如 main_script.m 调用 helper_script.m)
   - `Function -> Function`: 函数调用其他函数

2. **DEFINES 关系**：
   - `Script -> Variable`: 脚本定义变量 (如 main_script.m 定义 x, y, z)
   - `Function -> Variable`: 函数定义参数或局部变量 (如 modify_variables 定义 in1, out1)
   - `Script -> Function`: 脚本定义函数 (如脚本文件中定义的函数)

3. **USES 关系**：
   - `Script -> Variable`: 脚本使用外部变量 (如 main_script.m 使用 x, y)
   - `Function -> Variable`: 函数使用外部变量 (如 modify_variables 使用 in1)

4. **MODIFIES 关系**：
   - `Script -> Variable`: 脚本修改变量 (如 main_script.m 修改 x, y)
   - `Function -> Variable`: 函数修改变量

5. **ASSIGNED_TO 关系**：
   - `Variable -> Variable`: 变量赋值给其他变量 (如 x 赋值给 y, y 赋值给 z)

**变量作用域处理说明：**

**重要改进：每个脚本/函数中的变量都单独生成节点，即使变量名相同**

1. **变量ID格式**：`var_{变量名}_{作用域ID}`
   - 例如：`var_x_script_main_script` 表示 main_script.m 中的变量 x
   - 例如：`var_x_func_modify_variables` 表示 modify_variables 函数中的变量 x

2. **作用域隔离**：
   - 不同脚本中的同名变量被视为不同的节点
   - 不同函数中的同名变量被视为不同的节点
   - 每个变量节点包含 `scope_id` 和 `scope_type` 属性

3. **MATLAB实际语义**：
   - **脚本间变量隔离**：MATLAB脚本之间默认不能直接共享变量，除非变量已在基础工作区存在
   - **函数参数传递**：函数通过参数列表接收变量，这不是跨作用域访问，而是明确的参数传递
   - **工作区隔离**：每个脚本/函数有独立的工作区，变量不会自动跨作用域共享
   - **全局变量**：只有使用 `global` 关键字声明的变量才能在多个作用域间共享

4. **执行顺序分析**：
   - **自动检测脚本依赖关系**：通过分析 `script_name;` 和 `run('script_name.m')` 调用
   - **拓扑排序确定执行顺序**：确保被调用的脚本在调用者之前执行
   - **跨作用域变量访问**：只有在执行顺序上先定义的变量才能被后续脚本访问
   - **时序依赖关系**：创建带有执行顺序信息的跨作用域 `USES` 关系

**实际案例说明：**

1. **同变量名在不同作用域**：
   - `main_script.m` 中定义变量 `x` → 节点 `var_x_script_main_script`
   - `modify_variables` 函数中也有变量 `x` → 节点 `var_x_func_modify_variables`
   - 这两个变量是完全独立的节点
   - **跨作用域变量传递**：当脚本A中的变量被脚本B使用时，会生成：
     - `script_B -[USES]-> var_x_script_A` (使用关系)
     - `var_x_script_A -[ASSIGNED_TO]-> var_x_script_B` (赋值关系，体现依赖)

2. **函数参数传递（不是跨作用域访问）**：
   - `main_script.m` 调用 `modify_variables(x, y)` 时
   - 这是参数传递，不是跨作用域访问
   - 创建关系：`script_main_script -[CALLS]-> func_modify_variables`
   - 函数内部参数：`var_in1_func_modify_variables` 和 `var_in2_func_modify_variables`

3. **同作用域内变量使用**：
   - 在 `main_script.m` 中：`y = x + 5`
   - 创建关系：`script_main_script -[USES]-> var_x_script_main_script`
   - 创建关系：`var_x_script_main_script -[ASSIGNED_TO]-> var_y_script_main_script`

4. **函数内部变量处理**：
   - `modify_variables` 函数内部：`out1 = in1 * 2`
   - 创建关系：`var_in1_func_modify_variables -[ASSIGNED_TO]-> var_out1_func_modify_variables`

5. **全局变量处理**：
   - 如果使用 `global shared_var` 声明
   - 创建关系：`script_main_script -[DEFINES]-> var_shared_var_script_main_script` (type: "global_variable")

6. **执行顺序相关的跨作用域访问**：
   - `script1.m` 调用 `script2;` 然后使用 `val_from_script2`
   - 执行顺序：`script2.m` → `script1.m`
   - 创建关系：`script_script1 -[CALLS]-> script_script2`
   - 创建关系：`script_script1 -[USES]-> var_val_from_script2_script_script2` (usage_type: "cross_scope_execution_order")

**优势：**
- 符合MATLAB实际语义
- 自动分析脚本执行顺序
- 正确处理跨作用域变量访问
- 避免错误的跨作用域关系
- 清晰的作用域边界
- 便于可视化展示
- 支持正确的变量依赖分析

"""


EXAMPLES = """
```json
{
  "nodes": [
    {
      "type": "Script",
      "id": "script_main_script",
      "properties": {
        "name": "main_script.m",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "description": "Main script that demonstrates various relationship patterns"
      }
    },
    {
      "type": "Script",
      "id": "script_helper_script",
      "properties": {
        "name": "helper_script.m",
        "file_path": "tests/matlab_test/examples/helper_script.m",
        "description": "Helper script called by main script"
      }
    },
    {
      "type": "Function",
      "id": "func_helper_function",
      "properties": {
        "name": "helper_function",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "parameters": "input_value",
        "returns": "result",
        "description": "Helper function that processes input"
      }
    },
    {
      "type": "Function",
      "id": "func_modify_variables",
      "properties": {
        "name": "modify_variables",
        "file_path": "tests/matlab_test/examples/modify_variables.m",
        "parameters": "in1, in2",
        "returns": "out1, out2",
        "description": "Function that modifies its input variables"
      }
    },
    {
      "type": "Variable",
      "id": "var_x_script_main_script",
      "properties": {
        "name": "x",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "scope_id": "script_main_script",
        "scope_type": "script",
        "line_range": [["main script", 8, 8]]
      }
    },
    {
      "type": "Variable",
      "id": "var_y_script_main_script",
      "properties": {
        "name": "y",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "scope_id": "script_main_script",
        "scope_type": "script",
        "line_range": [["main script", 9, 9]]
      }
    },
    {
      "type": "Variable",
      "id": "var_z_script_main_script",
      "properties": {
        "name": "z",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "scope_id": "script_main_script",
        "scope_type": "script",
        "line_range": [["main script", 10, 10]]
      }
    },
    {
      "type": "Variable",
      "id": "var_result1_script_main_script",
      "properties": {
        "name": "result1",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "scope_id": "script_main_script",
        "scope_type": "script",
        "line_range": [["main script", 5, 5]]
      }
    },
    {
      "type": "Variable",
      "id": "var_in1_func_modify_variables",
      "properties": {
        "name": "in1",
        "file_path": "tests/matlab_test/examples/modify_variables.m",
        "scope_id": "func_modify_variables",
        "scope_type": "function",
        "line_range": [["function modify_variables", 2, 2]]
      }
    },
    {
      "type": "Variable",
      "id": "var_out1_func_modify_variables",
      "properties": {
        "name": "out1",
        "file_path": "tests/matlab_test/examples/modify_variables.m",
        "scope_id": "func_modify_variables",
        "scope_type": "function",
        "line_range": [["function modify_variables", 3, 3]]
      }
    },
    {
      "type": "Variable",
      "id": "var_x_func_modify_variables",
      "properties": {
        "name": "x",
        "file_path": "tests/matlab_test/examples/modify_variables.m",
        "scope_id": "func_modify_variables",
        "scope_type": "function",
        "line_range": [["function modify_variables", 3, 3]]
      }
    },
    {
      "type": "Variable",
      "id": "var_y_func_modify_variables",
      "properties": {
        "name": "y",
        "file_path": "tests/matlab_test/examples/modify_variables.m",
        "scope_id": "func_modify_variables",
        "scope_type": "function",
        "line_range": [["function modify_variables", 4, 4]]
      }
    }
  ],
  "relationships": [
    {
      "type": "CALLS",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "func_helper_function",
      "target_type": "Function",
      "properties": {
        "call_type": "function_call",
        "line_number": 5
      }
    },
    {
      "type": "CALLS",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "script_helper_script",
      "target_type": "Script",
      "properties": {
        "call_type": "script_call",
        "line_number": 8
      }
    },
    {
      "type": "CALLS",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "func_modify_variables",
      "target_type": "Function",
      "properties": {
        "call_type": "function_call",
        "line_number": 13
      }
    },
    {
      "type": "DEFINES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_x_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_definition",
        "line_number": 8
      }
    },
    {
      "type": "DEFINES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_y_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_definition",
        "line_number": 9
      }
    },
    {
      "type": "DEFINES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_z_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_definition",
        "line_number": 10
      }
    },
    {
      "type": "DEFINES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_result1_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_definition",
        "line_number": 5
      }
    },
    {
      "type": "DEFINES",
      "source_id": "func_modify_variables",
      "source_type": "Function",
      "target_id": "var_in1_func_modify_variables",
      "target_type": "Variable",
      "properties": {
        "type": "parameter",
        "line_number": 2
      }
    },
    {
      "type": "DEFINES",
      "source_id": "func_modify_variables",
      "source_type": "Function",
      "target_id": "var_out1_func_modify_variables",
      "target_type": "Variable",
      "properties": {
        "type": "local",
        "line_number": 3
      }
    },
    {
      "type": "USES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_x_script_main_script",
      "target_type": "Variable",
      "properties": {
        "usage_type": "local_variable",
        "variable_name": "x",
        "line_number": 9
      }
    },
    {
      "type": "USES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_y_script_main_script",
      "target_type": "Variable",
      "properties": {
        "usage_type": "local_variable",
        "variable_name": "y",
        "line_number": 10
      }
    },
    {
      "type": "USES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_config_script_setup_script",
      "target_type": "Variable",
      "properties": {
        "usage_type": "cross_scope_execution_order",
        "variable_name": "config",
        "source_script": "setup_script",
        "target_script": "main_script",
        "execution_order": 1,
        "source_execution_order": 0,
        "post_processed": true
      }
    },
    {
      "type": "ASSIGNED_TO",
      "source_id": "var_config_script_setup_script",
      "source_type": "Variable",
      "target_id": "var_config_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "cross_scope_assignment",
        "variable_name": "config",
        "source_script": "setup_script",
        "target_script": "main_script",
        "execution_order": 1,
        "source_execution_order": 0,
        "post_processed": true
      }
    },
    {
      "type": "MODIFIES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_x_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_modification",
        "line_number": 13
      }
    },
    {
      "type": "MODIFIES",
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "var_y_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_modification",
        "line_number": 13
      }
    },
    {
      "type": "ASSIGNED_TO",
      "source_id": "var_x_script_main_script",
      "source_type": "Variable",
      "target_id": "var_y_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_assignment",
        "line_number": 9
      }
    },
    {
      "type": "ASSIGNED_TO",
      "source_id": "var_y_script_main_script",
      "source_type": "Variable",
      "target_id": "var_z_script_main_script",
      "target_type": "Variable",
      "properties": {
        "type": "variable_assignment",
        "line_number": 10
      }
    }
  ]
}
```
"""
