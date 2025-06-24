#!/usr/bin/env python3
"""
测试启动脚本驱动的MATLAB跨作用域关系处理器

这个脚本用于验证：
1. 启动脚本执行顺序的正确分析
2. 基于执行顺序的跨作用域变量调用关系
3. 变量作用域时间线的正确构建
"""

import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from neo4j_graphrag.experimental.components.code_extractor.matlab.entry_script_driven_processor import EntryScriptDrivenProcessor
from neo4j_graphrag.experimental.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship

def create_test_files():
    """创建测试文件"""
    test_dir = Path("test_entry_script_data")
    test_dir.mkdir(exist_ok=True)

    # 启动脚本
    entry_content = """global_var = 10;
script_a;
script_b;
result = processed_data + global_var;"""

    # 脚本A
    script_a_content = """input_data = [1, 2, 3];
adjusted_data = input_data + global_var;
processed_data = helper_function(adjusted_data);"""

    # 脚本B
    script_b_content = """if exist('processed_data', 'var')
    final_result = processed_data * 1.5;
end
output_data = final_result;"""

    (test_dir / "main_script.m").write_text(entry_content, encoding='utf-8')
    (test_dir / "script_a.m").write_text(script_a_content, encoding='utf-8')
    (test_dir / "script_b.m").write_text(script_b_content, encoding='utf-8')

    return test_dir

def create_test_graph():
    """创建测试用的图结构"""
    # 创建脚本节点
    main_script = Neo4jNode(
        id="script_main_script_test_entry_script_data/main_script.m",
        label="Script",
        properties={
            'name': 'main_script',
            'file_path': 'test_entry_script_data/main_script.m',
            'code_snippet': """% 启动脚本 - 测试跨作用域变量调用
% 定义全局变量
global_var = 10;
config_param = 'test_config';

% 调用第一个脚本
script_a;

% 调用第二个脚本
script_b;

% 使用脚本b中定义的变量
result = processed_data + global_var;
disp(['Final result: ' num2str(result)]);"""
        }
    )

    script_a = Neo4jNode(
        id="script_script_a_test_entry_script_data/script_a.m",
        label="Script",
        properties={
            'name': 'script_a',
            'file_path': 'test_entry_script_data/script_a.m',
            'code_snippet': """% 脚本A - 定义一些变量
input_data = [1, 2, 3, 4, 5];
temp_var = input_data * 2;

% 使用启动脚本中的变量
adjusted_data = input_data + global_var;

% 调用辅助函数
processed_data = helper_function(adjusted_data);"""
        }
    )

    script_b = Neo4jNode(
        id="script_script_b_test_entry_script_data/script_b.m",
        label="Script",
        properties={
            'name': 'script_b',
            'file_path': 'test_entry_script_data/script_b.m',
            'code_snippet': """% 脚本B - 使用脚本A中的变量
% 使用脚本A中定义的变量
if exist('processed_data', 'var')
    final_result = processed_data * 1.5;
else
    final_result = 0;
end

% 定义新变量供后续脚本使用
output_data = final_result;"""
        }
    )

    helper_function = Neo4jNode(
        id="function_helper_function_test_entry_script_data/helper_function.m",
        label="Function",
        properties={
            'name': 'helper_function',
            'file_path': 'test_entry_script_data/helper_function.m',
            'code_snippet': """function result = helper_function(input_data)
% 辅助函数 - 处理输入数据
result = sum(input_data) + config_param;
end"""
        }
    )

    # 创建变量节点
    global_var = Neo4jNode(
        id="var_global_var_script_main_script",
        label="Variable",
        properties={
            'name': 'global_var',
            'type': 'local',
            'scope_id': 'main_script',
            'scope_type': 'script'
        }
    )

    processed_data = Neo4jNode(
        id="var_processed_data_script_script_a",
        label="Variable",
        properties={
            'name': 'processed_data',
            'type': 'local',
            'scope_id': 'script_a',
            'scope_type': 'script'
        }
    )

    # 创建图
    graph = Neo4jGraph(
        nodes=[main_script, script_a, script_b, helper_function, global_var, processed_data],
        relationships=[]
    )

    return graph

async def test_processor():
    """测试处理器"""
    print("=== 测试启动脚本驱动的处理器 ===\n")

    # 创建测试文件
    test_dir = create_test_files()
    entry_script_path = str(test_dir / "main_script.m")

    # 初始化处理器
    processor = EntryScriptDrivenProcessor()
    processor.set_entry_script(entry_script_path)

    print(f"启动脚本: {entry_script_path}")
    print(f"执行顺序: {processor.script_execution_order}")

    # 注册脚本变量
    for file_path in test_dir.glob("*.m"):
        if file_path.name != "main_script.m":
            script_name = file_path.stem
            content = file_path.read_text(encoding='utf-8')
            processor.register_script_variables(script_name, content)

    print(f"脚本A变量定义: {list(processor.variable_definitions.get('script_a', {}).keys())}")
    print(f"脚本B变量定义: {list(processor.variable_definitions.get('script_b', {}).keys())}")

    # 创建测试图
    nodes = [
        Neo4jNode(id="script_main_script_test", label="Script",
                 properties={'name': 'main_script', 'code_snippet': 'test'}),
        Neo4jNode(id="script_script_a_test", label="Script",
                 properties={'name': 'script_a', 'code_snippet': 'test'}),
        Neo4jNode(id="script_script_b_test", label="Script",
                 properties={'name': 'script_b', 'code_snippet': 'test'})
    ]

    test_graph = Neo4jGraph(nodes=nodes, relationships=[])

    # 处理跨作用域关系
    final_graph = processor.process_cross_scope_relationships(test_graph)

    print(f"最终关系数: {len(final_graph.relationships)}")

    # 清理
    import shutil
    shutil.rmtree(test_dir)
    print("测试完成")

if __name__ == "__main__":
    asyncio.run(test_processor())
