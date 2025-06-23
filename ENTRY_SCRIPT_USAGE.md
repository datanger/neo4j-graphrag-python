# MATLAB 启动脚本功能使用指南

## 概述

MATLAB 启动脚本功能解决了跨作用域变量识别的问题。通过指定一个启动脚本，系统可以分析脚本的执行流程，准确识别变量的来源和作用域。

## 功能特性

### 1. 启动脚本分析
- 自动分析启动脚本中的脚本调用顺序
- 识别启动脚本中定义的变量
- 建立基于执行顺序的变量作用域关系

### 2. 跨作用域变量识别
- 根据启动脚本的执行流程确定变量来源
- 避免同名变量在不同脚本中的混淆
- 支持 `script_name;` 和 `run('script_name.m')` 调用方式

### 3. 灵活的数据构建模式
- **完整重建模式**: 重新构建所有数据
- **增量更新模式**: 只更新后处理，重用现有数据
- **自动检测**: 自动判断是否需要重建数据

## 使用方法

### 命令行参数

```bash
python examples/customize/build_graph/pipeline/kg_builder_from_code.py [选项]
```

#### 主要参数

- `--matlab_dir`: MATLAB文件目录 (默认: "tests/matlab_test/examples")
- `--entry_script`: 启动脚本路径 (可选)
- `--rebuild_data`: 重新构建所有数据 (默认: False)
- `--neo4j_uri`: Neo4j连接URI (默认: "bolt://localhost:7687")
- `--neo4j_user`: Neo4j用户名 (默认: "neo4j")
- `--neo4j_password`: Neo4j密码 (默认: "password")
- `--neo4j_db`: Neo4j数据库名 (默认: "neo4j")

### 使用场景

#### 1. 不使用启动脚本（原有方式）

```bash
python examples/customize/build_graph/pipeline/kg_builder_from_code.py \
    --matlab_dir /path/to/matlab/files
```

这种方式使用拓扑排序和依赖分析来确定脚本执行顺序。

#### 2. 使用启动脚本进行完整重建

```bash
python examples/customize/build_graph/pipeline/kg_builder_from_code.py \
    --matlab_dir /path/to/matlab/files \
    --entry_script /path/to/main.m \
    --rebuild_data
```

#### 3. 使用启动脚本进行增量更新

```bash
python examples/customize/build_graph/pipeline/kg_builder_from_code.py \
    --matlab_dir /path/to/matlab/files \
    --entry_script /path/to/main.m
```

如果已有构建数据，只更新后处理；如果没有数据，会从头开始构建。

#### 4. 针对大型项目（如transformer-models）

```bash
python examples/customize/build_graph/pipeline/kg_builder_from_code.py \
    --matlab_dir /home/niejie/work/Code/tools/transformer-models \
    --entry_script /home/niejie/work/Code/tools/transformer-models/main.m
```

## 启动脚本格式

启动脚本应该包含以下内容：

```matlab
% 启动脚本示例 - main.m

% 定义全局变量
global_config = 'production_config';
data_path = '/path/to/data';

% 调用其他脚本（按执行顺序）
setup_environment;
process_data;
visualize_results;
```

### 支持的调用方式

1. **直接调用**: `script_name;`
2. **run函数调用**: `run('script_name.m');`

### 变量定义和使用

- 在启动脚本中定义的变量可以被后续脚本访问
- 系统会根据执行顺序自动识别变量来源
- 支持全局变量和局部变量

## 技术实现

### 1. 全局注册表增强

`GlobalMatlabRegistry` 类新增了以下功能：

```python
class GlobalMatlabRegistry:
    def set_entry_script(self, script_path: str, script_content: str = None):
        """设置启动脚本"""
        
    def _analyze_entry_script_execution_flow(self):
        """分析启动脚本执行流程"""
        
    def get_entry_script_execution_order(self) -> List[str]:
        """获取基于启动脚本的执行顺序"""
        
    def get_entry_script_variable_source(self, var_name: str, usage_script: str) -> Optional[str]:
        """根据启动脚本确定变量来源"""
```

### 2. 提取器增强

`MatlabExtractor` 类新增了以下功能：

```python
class MatlabExtractor:
    def __init__(self, ..., entry_script_path: Optional[str] = None):
        """支持启动脚本路径参数"""
        
    async def run(self, ..., rebuild_data: bool = False):
        """支持重建数据选项"""
```

### 3. 后处理增强

新增了专门处理启动脚本的后处理方法：

```python
@classmethod
def _process_entry_script_variable_access(cls, graph: Neo4jGraph):
    """处理启动脚本的跨作用域变量访问"""
```

## 示例

### 测试脚本

运行测试脚本来验证功能：

```bash
python test_entry_script.py
```

### 实际项目使用

对于 `/home/niejie/work/Code/tools/transformer-models` 项目：

1. **首次构建**:
```bash
python examples/customize/build_graph/pipeline/kg_builder_from_code.py \
    --matlab_dir /home/niejie/work/Code/tools/transformer-models \
    --entry_script /home/niejie/work/Code/tools/transformer-models/main.m
```

2. **更新启动脚本后重新分析**:
```bash
python examples/customize/build_graph/pipeline/kg_builder_from_code.py \
    --matlab_dir /home/niejie/work/Code/tools/transformer-models \
    --entry_script /home/niejie/work/Code/tools/transformer-models/main.m
```

3. **强制重新构建**:
```bash
python examples/customize/build_graph/pipeline/kg_builder_from_code.py \
    --matlab_dir /home/niejie/work/Code/tools/transformer-models \
    --entry_script /home/niejie/work/Code/tools/transformer-models/main.m \
    --rebuild_data
```

## 优势

1. **准确性**: 基于实际执行流程，避免变量来源混淆
2. **效率**: 支持增量更新，避免重复构建
3. **灵活性**: 支持多种使用模式
4. **可扩展性**: 易于扩展到其他编程语言

## 注意事项

1. 启动脚本应该是项目的实际入口点
2. 脚本调用顺序应该反映实际的执行逻辑
3. 变量命名应该避免冲突
4. 大型项目建议使用启动脚本来提高准确性

## 故障排除

### 常见问题

1. **启动脚本未找到**: 检查文件路径是否正确
2. **变量来源无法确定**: 检查启动脚本中的调用顺序
3. **执行顺序错误**: 确保启动脚本反映了实际的执行逻辑

### 调试信息

启用详细日志来查看分析过程：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 未来扩展

1. 支持更复杂的执行流程分析
2. 支持条件分支和循环结构
3. 扩展到其他编程语言
4. 支持多启动脚本 