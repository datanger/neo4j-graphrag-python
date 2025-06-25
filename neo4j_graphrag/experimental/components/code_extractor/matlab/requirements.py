from neo4j_graphrag.experimental.components.schema import GraphSchema, NodeType, PropertyType, RelationshipType

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
                PropertyType(name="scope_id", type="STRING", description="ID of the scope (script, function, or class) where this variable is defined"),
                PropertyType(name="scope_type", type="STRING", description="Type of scope: 'script', 'function', or 'class'"),
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
        NodeType(
            label="Class",
            description="A MATLAB class definition using classdef",
            properties=[
                PropertyType(name="name", type="STRING", description="Name of the class"),
                PropertyType(name="file_path", type="STRING", description="Path to the file containing the class definition"),
                PropertyType(name="line_range", type="STRING", description="Line range where the class is defined"),
                PropertyType(name="description", type="STRING", description="Class description from docstring"),
                PropertyType(name="superclass", type="STRING", description="Name of the superclass if the class inherits from another class"),
                PropertyType(name="scope_id", type="STRING", description="ID of the scope (script) where this class is defined"),
                PropertyType(name="scope_type", type="STRING", description="Type of scope: 'script' (classes are always defined in scripts)"),
            ],
        ),
    ],
    relationship_types=[
        RelationshipType(
            label="CALLS",
            description="A function, script, or class method calls another function or script"
        ),
        RelationshipType(
            label="USES",
            description="A function, script, or class method uses a variable which is defined in another function, script, or class",
        ),
        RelationshipType(
            label="DEFINES",
            description="A script defines a variable, function, or class; a function defines a variable; a class defines properties, methods, or nested classes",
        ),
        RelationshipType(
            label="ASSIGNED_TO",
            description="A variable is assigned to another variable",
        ),
        RelationshipType(
            label="MODIFIES",
            description="A function, script, or class method modifies a variable that was defined in another scope",
        ),
        RelationshipType(
            label="INHERITS_FROM",
            description="A class inherits from another class",
        ),
        RelationshipType(
            label="HAS_METHOD",
            description="A class has a method (function defined within the class)",
        ),
        RelationshipType(
            label="HAS_PROPERTY",
            description="A class has a property (variable defined within the class)",
        ),
    ],
    patterns=[
        ("Function", "CALLS", "Function"),
        ("Script", "CALLS", "Function"),
        ("Script", "CALLS", "Script"),
        ("Class", "CALLS", "Function"),
        ("Class", "CALLS", "Script"),
        ("Function", "USES", "Variable"),
        ("Script", "USES", "Variable"),
        ("Class", "USES", "Variable"),
        ("Function", "DEFINES", "Variable"),
        ("Script", "DEFINES", "Variable"),
        ("Script", "DEFINES", "Function"),
        ("Script", "DEFINES", "Class"),
        ("Class", "DEFINES", "Variable"),
        ("Script", "MODIFIES", "Variable"),
        ("Class", "MODIFIES", "Variable"),
        ("Variable", "ASSIGNED_TO", "Variable"),
        ("Class", "INHERITS_FROM", "Class"),
        ("Class", "HAS_METHOD", "Function"),
        ("Class", "HAS_PROPERTY", "Variable"),
    ],
)


SCHEMA_DESCRIPTION = """
**关系模式说明：**

1. **CALLS 关系**：
   - `Script -> Function`: 脚本调用函数 (如 main_script.m 调用 helper_function)
   - `Script -> Script`: 脚本调用其他脚本 (如 main_script.m 调用 helper_script.m)
   - `Function -> Function`: 函数调用其他函数
   - `Class -> Function`: 类方法调用函数 (如 MyClass.method 调用 helper_function)
   - `Class -> Script`: 类方法调用脚本 (如 MyClass.method 调用 helper_script.m)

2. **DEFINES 关系**：
   - `Script -> Variable`: 脚本定义变量 (如 main_script.m 定义 x, y, z)
   - `Function -> Variable`: 函数定义参数或局部变量 (如 modify_variables 定义 in1, out1)
   - `Script -> Function`: 脚本定义函数 (如脚本文件中定义的函数)
   - `Script -> Class`: 脚本定义类 (如脚本文件中定义的 classdef)
   - `Class -> Variable`: 类定义属性 (如 MyClass 定义 property1, property2)

3. **USES 关系**：
   - `Script -> Variable`: 脚本使用外部变量 (如 main_script.m 使用 x, y)
   - `Function -> Variable`: 函数使用外部变量 (如 modify_variables 使用 in1)
   - `Class -> Variable`: 类方法使用变量 (如 MyClass.method 使用 property1)

4. **MODIFIES 关系**：
   - `Script -> Variable`: 脚本修改变量 (如 main_script.m 修改 x, y)
   - `Function -> Variable`: 函数修改变量
   - `Class -> Variable`: 类方法修改变量 (如 MyClass.method 修改 property1)

5. **ASSIGNED_TO 关系**：
   - `Variable -> Variable`: 变量赋值给其他变量 (如 x 赋值给 y, y 赋值给 z)

6. **INHERITS_FROM 关系**：
   - `Class -> Class`: 类继承自另一个类 (如 MyClass < BaseClass)

7. **HAS_METHOD 关系**：
   - `Class -> Function`: 类拥有方法 (如 MyClass 拥有 method1, method2)

8. **HAS_PROPERTY 关系**：
   - `Class -> Variable`: 类拥有属性 (如 MyClass 拥有 property1, property2)

**变量作用域处理说明：**

**重要改进：每个脚本/函数/类中的变量都单独生成节点，即使变量名相同**

1. **变量ID格式**：`var_{变量名}_{作用域ID}`
   - 例如：`var_x_script_main_script` 表示 main_script.m 中的变量 x
   - 例如：`var_x_func_modify_variables` 表示 modify_variables 函数中的变量 x
   - 例如：`var_x_class_MyClass` 表示 MyClass 类中的变量 x

2. **作用域隔离**：
   - 不同脚本中的同名变量被视为不同的节点
   - 不同函数中的同名变量被视为不同的节点
   - 不同类中的同名变量被视为不同的节点
   - 每个变量节点包含 `scope_id` 和 `scope_type` 属性

3. **MATLAB实际语义**：
   - **脚本间变量隔离**：MATLAB脚本之间默认不能直接共享变量，除非变量已在基础工作区存在
   - **函数参数传递**：函数通过参数列表接收变量，这不是跨作用域访问，而是明确的参数传递
   - **工作区隔离**：每个脚本/函数有独立的工作区，变量不会自动跨作用域共享
   - **类属性隔离**：每个类实例有独立的属性空间
   - **全局变量**：只有使用 `global` 关键字声明的变量才能在多个作用域间共享

4. **执行顺序分析**：
   - **自动检测脚本依赖关系**：通过分析 `script_name;` 和 `run('script_name.m')` 调用
   - **拓扑排序确定执行顺序**：确保被调用的脚本在调用者之前执行
   - **跨作用域变量访问**：只有在执行顺序上先定义的变量才能被后续脚本访问
   - **时序依赖关系**：创建带有执行顺序信息的跨作用域 `USES` 关系

**Classdef 特殊处理说明：**

1. **类定义位置**：
   - 类总是定义在脚本文件中（.m文件）
   - 类定义使用 `classdef` 关键字
   - 类可以继承自其他类：`classdef MyClass < BaseClass`

2. **类结构**：
   - **properties 块**：定义类的属性（变量）
   - **methods 块**：定义类的方法（函数）
   - **events 块**：定义类的事件
   - **enumeration 块**：定义类的枚举

3. **类节点生成**：
   - 每个 `classdef` 生成一个 Class 节点
   - 类节点包含继承信息（superclass 属性）
   - 类的作用域是定义它的脚本

4. **类属性处理**：
   - properties 块中的每个属性生成一个 Variable 节点
   - 属性节点的 scope_type 为 'class'
   - 属性节点与类节点通过 HAS_PROPERTY 关系连接

5. **类方法处理**：
   - methods 块中的每个方法生成一个 Function 节点
   - 方法节点的 scope_type 为 'class'
   - 方法节点与类节点通过 HAS_METHOD 关系连接

6. **继承关系**：
   - 如果类继承自其他类，创建 INHERITS_FROM 关系
   - 继承关系指向父类节点

**实际案例说明：**

1. **类定义示例**：
   ```matlab
   classdef MyClass < BaseClass
       properties
           property1
           property2
       end
       
       methods
           function obj = MyClass()
               obj.property1 = 1;
           end
           
           function result = method1(obj, input)
               result = obj.property1 + input;
           end
       end
   end
   ```

2. **生成的节点和关系**：
   - Class 节点：`class_MyClass_script_main_script`
   - Variable 节点：`var_property1_class_MyClass`, `var_property2_class_MyClass`
   - Function 节点：`func_MyClass_class_MyClass`, `func_method1_class_MyClass`
   - 关系：
     - `class_MyClass -[INHERITS_FROM]-> class_BaseClass`
     - `class_MyClass -[HAS_PROPERTY]-> var_property1_class_MyClass`
     - `class_MyClass -[HAS_PROPERTY]-> var_property2_class_MyClass`
     - `class_MyClass -[HAS_METHOD]-> func_MyClass_class_MyClass`
     - `class_MyClass -[HAS_METHOD]-> func_method1_class_MyClass`

3. **同变量名在不同作用域**：
   - `main_script.m` 中定义变量 `x` → 节点 `var_x_script_main_script`
   - `modify_variables` 函数中也有变量 `x` → 节点 `var_x_func_modify_variables`
   - `MyClass` 类中也有属性 `x` → 节点 `var_x_class_MyClass`
   - 这三个变量是完全独立的节点

4. **类方法调用**：
   - `MyClass.method1()` 调用 `helper_function()`
   - 创建关系：`func_method1_class_MyClass -[CALLS]-> func_helper_function`

5. **类属性使用**：
   - `MyClass.method1` 中使用 `obj.property1`
   - 创建关系：`func_method1_class_MyClass -[USES]-> var_property1_class_MyClass`

**优势：**
- 符合MATLAB实际语义
- 支持完整的面向对象编程结构
- 自动分析脚本执行顺序
- 正确处理跨作用域变量访问
- 避免错误的跨作用域关系
- 清晰的作用域边界
- 便于可视化展示
- 支持正确的变量依赖分析
- 支持类继承和方法调用分析

"""

REQUIREMENTS = """
1. 作用域的定义：一个脚本、函数或类内就是一个独立的作用域，这是为了隔离其他作用域的变量，避免变量污染。
2. 每个作用域内的变量都单独生成节点，即使变量名相同。
3. 作用域隔离：不同脚本中的同名变量被视为不同的节点，不同函数中的同名变量被视为不同的节点，不同类中的同名变量被视为不同的节点。
4. 跨作用域变量访问：只有在执行顺序上先定义的变量才能被后续脚本访问。此时应当生成跨作用域的USES关系，该作用域内被USES的这个变量与跨作用域的同名变量间应当同时有ASSIGNED_TO关系，以此来体现跨作用域的变量依赖关系。
5. 每个变量不能独立存在，应当与作用域绑定，要么是脚本中的变量，要么是函数中的变量，要么是类中的属性。
6. 每个函数也不能独立存在，函数的定义肯定是在脚本或类中，因此函数节点应当与脚本或类节点绑定。
7. 每个类也不能独立存在，类的定义肯定是在脚本中，因此类节点应当与脚本节点绑定。
8. 类继承关系：如果类继承自其他类，应当创建INHERITS_FROM关系。
9. 类属性和方法：类的properties块中的每个属性应当生成Variable节点，methods块中的每个方法应当生成Function节点，并通过HAS_PROPERTY和HAS_METHOD关系与类节点连接。
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
      "type": "Class",
      "id": "class_MyClass_script_main_script",
      "properties": {
        "name": "MyClass",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "line_range": "15-25",
        "description": "Example class with properties and methods",
        "superclass": "BaseClass",
        "scope_id": "script_main_script",
        "scope_type": "script"
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
    },
    {
      "type": "Variable",
      "id": "var_property1_class_MyClass",
      "properties": {
        "name": "property1",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "scope_id": "class_MyClass_script_main_script",
        "scope_type": "class",
        "line_range": [["class MyClass", 16, 16]]
      }
    },
    {
      "type": "Function",
      "id": "func_method1_class_MyClass",
      "properties": {
        "name": "method1",
        "file_path": "tests/matlab_test/examples/main_script.m",
        "parameters": "obj, input",
        "returns": "result",
        "description": "Class method that processes input",
        "scope_id": "class_MyClass_script_main_script",
        "scope_type": "class"
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
      "source_id": "script_main_script",
      "source_type": "Script",
      "target_id": "class_MyClass_script_main_script",
      "target_type": "Class",
      "properties": {
        "type": "class_definition",
        "line_number": 15
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
    },
    {
      "type": "INHERITS_FROM",
      "source_id": "class_MyClass_script_main_script",
      "source_type": "Class",
      "target_id": "class_BaseClass_script_base_script",
      "target_type": "Class",
      "properties": {
        "type": "class_inheritance",
        "line_number": 15
      }
    },
    {
      "type": "HAS_PROPERTY",
      "source_id": "class_MyClass_script_main_script",
      "source_type": "Class",
      "target_id": "var_property1_class_MyClass",
      "target_type": "Variable",
      "properties": {
        "type": "class_property",
        "line_number": 16
      }
    },
    {
      "type": "HAS_METHOD",
      "source_id": "class_MyClass_script_main_script",
      "source_type": "Class",
      "target_id": "func_method1_class_MyClass",
      "target_type": "Function",
      "properties": {
        "type": "class_method",
        "line_number": 20
      }
    },
    {
      "type": "CALLS",
      "source_id": "func_method1_class_MyClass",
      "source_type": "Function",
      "target_id": "func_helper_function",
      "target_type": "Function",
      "properties": {
        "call_type": "function_call",
        "line_number": 22
      }
    },
    {
      "type": "USES",
      "source_id": "func_method1_class_MyClass",
      "source_type": "Function",
      "target_id": "var_property1_class_MyClass",
      "target_type": "Variable",
      "properties": {
        "usage_type": "class_property",
        "variable_name": "property1",
        "line_number": 21
      }
    }
  ]
}
```
"""