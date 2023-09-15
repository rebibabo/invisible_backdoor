import os
import sys
import shutil
from tqdm import tqdm
sys.path.append('attack/ropgen')
from itertools import combinations
from aug_data.change_program_style import *
from style_change_method import var_name_style_to_camel_case, \
                                var_name_style_to_initcap, \
                                var_name_style_to_init_underscore, \
                                var_name_style_to_underscore, \
                                var_name_style_to_init_dollar, \
                                cpp_lib_to_c_2, \
                                cpp_lib_to_c_3, \
                                incr_opr_prepost_to_incr_postfix, \
                                incr_opr_prepost_to_incr_prefix, \
                                incr_opr_prepost_to_full_incr, \
                                incr_opr_prepost_to_separate_incr

# 2,3,4,12,13,14,15,16,17 need target author
# 18 to cpp
style_mapping = {
    '1.1': 'var_name_style_to_camel_case',
    '1.2': 'var_name_style_to_initcap',
    '1.3': 'var_name_style_to_underscore',
    '1.4': 'var_name_style_to_init_underscore',
    '1.5': 'var_name_style_to_init_dollar',
    # 5.1 数据集使用指针较少，conv -> 0%
    '5.1': 'pointer_to_array', 
    '5.2': 'array_to_pointer', 
    '6.1': 're_temp', # defined at the beginning
    '6.2': 'temporary_var',
    '7.1': 'var_init_merge',
    '7.2': 'var_init_pos',
    '8.1': 'var_init_split',
    '8.2': 'init_declaration',
    # 9.x 在数据集中都较为稀少，conv -> 0%
    '9.1': 'assign_value',  
    '9.2': 'assign_combine',
    # 10.x 能够提取for循环语句中的i++，但无法转化for循环语句中的i++
    '10.1': 'incr_opr_prepost_to_incr_postfix',
    '10.2': 'incr_opr_prepost_to_incr_prefix',
    '10.3': 'incr_opr_prepost_to_full_incr',
    '10.4': 'incr_opr_prepost_to_separate_incr', 
    # 11.x 训练集中全是函数，结构体很少，conv -> 0%
    '11.1': 'typedef',
    '11.2': 'retypedef',
    # 19.1 >> 19.2
    '19.1': 'dyn_static_mem',
    '19.2': 'static_dyn_mem',
    '20.1': 'while_for',
    '20.2': 'for_while',
    '21.1': 'switch_if',
    '21.2': 'ternary',
    '22.1': 'if_spilt',
    '22.2': 'if_combine'
}

def change_style_OR(code, choice):
    converted_styles = []
    for idx in choice:
        if idx in style_mapping:
            converted_styles.append(style_mapping[idx])
    if not os.path.exists('temp'):
        os.mkdir('temp')
    code_file = 'temp/code.c'
    copy_file = 'temp/copy.c'
    xml_file = 'temp/xml'
    code_change_file = 'temp/change.c'
    with open(code_file,'w') as f:
        f.write(code)
    shutil.copy(code_file, copy_file)
    for i in range(len(converted_styles)):
        get_style.srcml_program_xml(copy_file, xml_file)
        eval(converted_styles[i]).program_transform_save_div(xml_file, './')
        get_style.srcml_xml_program(xml_file + '.xml', code_change_file)
        shutil.move(code_change_file, copy_file)
    code = open(copy_file).read()
    succ = compare_files(code_file, copy_file)
    shutil.rmtree('temp')
    return code, succ

def change_style_AND(code, choice):
    converted_styles = []
    for idx in choice:
        if idx in style_mapping:
            converted_styles.append(style_mapping[idx])
    if not os.path.exists('temp'):
        os.mkdir('temp')
    code_file = 'temp/code.c'
    copy_file = 'temp/copy.c'
    xml_file = 'temp/xml'
    code_change_file = 'temp/change.c'
    with open(code_file,'w') as f:
        f.write(code)
    
    succ = 1
    for i in range(len(converted_styles)):
        shutil.copy(code_file, copy_file)
        get_style.srcml_program_xml(copy_file, xml_file)
        eval(converted_styles[i]).program_transform_save_div(xml_file, './')
        get_style.srcml_xml_program(xml_file + '.xml', code_change_file)
        shutil.move(code_change_file, copy_file)
        succ &= compare_files(code_file, copy_file)

    shutil.copy(code_file, copy_file)
    for i in range(len(converted_styles)):
        get_style.srcml_program_xml(copy_file, xml_file)
        eval(converted_styles[i]).program_transform_save_div(xml_file, './')
        get_style.srcml_xml_program(xml_file + '.xml', code_change_file)
        shutil.move(code_change_file, copy_file)
    shutil.rmtree('temp')
    return code, succ