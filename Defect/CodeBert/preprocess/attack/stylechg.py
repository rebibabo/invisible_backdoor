import os
import sys
import shutil
from tqdm import tqdm
sys.path.append('./ropgen')
# sys.path.append('/home/backdoor2023/backdoor/preprocess/attack/ropgen')
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
                                incr_opr_prepost_to_separate_incr, \
                                var_name_style_to_camel_case_ex, \
                                var_name_style_to_initcap_ex, \
                                var_name_style_to_init_underscore_ex, \
                                var_name_style_to_underscore_ex, \
                                var_name_style_to_init_dollar_ex, \
                                incr_opr_prepost_to_incr_postfix_ex, \
                                incr_opr_prepost_to_incr_prefix_ex, \
                                incr_opr_prepost_to_full_incr_ex, \
                                incr_opr_prepost_to_separate_incr_ex, \
                                var_name_style_to_allcap_ex
                                
from style_change_method.var_name_style_to_camel_case_ex import random_camel_case_variable_name
from style_change_method.var_name_style_to_initcap_ex import random_initcap

# 2,3,4,12,13,14,15,16,17 need target author
# 18 to cpp
style_mapping = {
    '1.1': 'var_name_style_to_camel_case',
    '1.2': 'var_name_style_to_initcap',
    '1.3': 'var_name_style_to_underscore',
    '1.4': 'var_name_style_to_init_underscore',
    '1.5': 'var_name_style_to_init_dollar',
    '1.1ex': 'var_name_style_to_camel_case_ex',
    '1.2ex': 'var_name_style_to_initcap_ex',
    '1.3ex': 'var_name_style_to_underscore_ex',
    '1.4ex': 'var_name_style_to_init_underscore_ex',
    '1.5ex': 'var_name_style_to_init_dollar_ex',
    '1.6ex': 'var_name_style_to_allcap_ex',
    # 5.1 数据集使用指针较少，conv -> 0%
    '5.1': 'pointer_to_array', 
    '5.2': 'array_to_pointer', 
    '6.1': 're_temp', # defined at the beginning
    '6.2': 'temporary_var',
    '7.1': 'var_init_merge',                        # int a = 0;
    '7.2': 'var_init_pos',                          # int a; a = 0;
    '8.1': 'init_declaration',
    '8.2': 'var_init_split',
    # 9.x 在数据集中都较为稀少，conv -> 0%
    '9.1': 'assign_value',  
    '9.2': 'assign_combine',
    '10.1': 'incr_opr_prepost_to_incr_postfix',      # i++ 
    '10.2': 'incr_opr_prepost_to_incr_prefix',       # ++i
    '10.3': 'incr_opr_prepost_to_full_incr',         # i=i+1
    '10.4': 'incr_opr_prepost_to_separate_incr',     # i+=1
    '10.1ex': 'incr_opr_prepost_to_incr_postfix_ex', # x++
    '10.2ex': 'incr_opr_prepost_to_incr_prefix_ex',  # ++x
    '10.3ex': 'incr_opr_prepost_to_full_incr_ex',    # x=x+1
    '10.4ex': 'incr_opr_prepost_to_separate_incr_ex',   # x+=1
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

def find_right_bracket(string):
    stack = []
    for index, char in enumerate(string):
        if char == '(':
            stack.append(char)
        elif char == ')':
            stack.pop()
            if len(stack) == 0:
                return index
    return -1 

def find_func_beginning(code):
    right_bracket = find_right_bracket(code)
    insert_index = code.find('{', right_bracket)
    return insert_index + 1

def style_expand(code, style):
    if style == '1.1ex':
        begin_index = find_func_beginning(code)
        insert_code = 'int ' + random_camel_case_variable_name(3) + ' = 1;' + \
                      'int ' + random_camel_case_variable_name(3) + ' = 1'
        code = code[:begin_index] + insert_code + code[begin_index:]
    elif style == '1.2ex':
        begin_index = find_func_beginning(code)
        insert_code = 'int ' + random_initcap(3) + ' = 1;'
        code = code[:begin_index] + insert_code + code[begin_index:]
    return code

def change_style_AND(code, choice, file_type='c', expand=False):
    converted_styles = []
    for idx in choice:
        if idx in style_mapping:
            converted_styles.append(style_mapping[idx])
    temp_dir = 'temp_' + '_'.join(choice)
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    code_file = temp_dir + '/code.' + file_type
    copy_file = temp_dir + '/copy.' + file_type
    xml_file = temp_dir + '/xml'
    code_change_file = temp_dir + '/change' + file_type
    res_code = []
    res_code.append(code)
    with open(code_file,'w') as f:
        f.write(code)
    get_style.srcml_program_xml(code_file, xml_file)
    program_style = get_style.get_style(xml_file + '.xml', file_type)
    succ = 1
    res = []
    for target_style in choice:
        if 'ex' in target_style:
            target_style = target_style[:-2]
        for d in program_style:
            if target_style in d:
                res.append(d)
                succ &= (max(d.values()) == d[target_style] and d[target_style] > 0)
                break
        if not succ:
            break
    if succ and '1.6ex' not in choice:
        return code, 1

    shutil.copy(code_file, copy_file)
    for i in range(len(converted_styles)):
        get_style.srcml_program_xml(copy_file, xml_file)
        if converted_styles[i] == style_mapping['1.6ex']:
            tsucc = eval(converted_styles[i]).program_transform_save_div(xml_file, './')
        else:
            eval(converted_styles[i]).program_transform_save_div(xml_file, './')
        get_style.srcml_xml_program(xml_file + '.xml', code_change_file)
        shutil.move(code_change_file, copy_file)
    get_style.srcml_program_xml(copy_file, xml_file)
    program_style = get_style.get_style(xml_file + '.xml', file_type)
    succ = 1
    for target_style in choice:
        if 'ex' in target_style:
            # 特殊判断：10.xex 修改循环语句中的变量
            if target_style.split('.')[0] == '10':
                continue
            target_style = target_style[:-2]
        for d in program_style:
            if target_style in d:
                res.append(d)
                succ &= (max(d.values()) == d[target_style] and d[target_style] > 0)
                break
        if not succ:
            break
    with open(copy_file, 'r') as f:
        code = f.read()
    res_code.append(code)
    shutil.rmtree(temp_dir)
    if '1.6ex' in choice:
        succ = tsucc
    if expand:
        code = style_expand(code, choice[0])
        return code, 1
    return code, succ

def change_style_OR(code, choice, file_type='c'):
    converted_styles = []
    for idx in choice:
        if idx in style_mapping:
            converted_styles.append(style_mapping[idx])
    temp_dir = 'temp_' + '_'.join(choice)
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    code_file = temp_dir + '/code.' + file_type
    copy_file = temp_dir + '/copy.' + file_type
    xml_file = temp_dir + '/xml'
    code_change_file = temp_dir + '/change' + file_type
    res_code = []
    res_code.append(code)
    with open(code_file,'w') as f:
        f.write(code)
    get_style.srcml_program_xml(code_file, xml_file)
    program_style = get_style.get_style(xml_file + '.xml', file_type)
    succ = 1
    res = []
    for target_style in choice:
        if 'ex' in target_style:
            target_style = target_style[:-2]
        for d in program_style:
            if target_style in d:
                res.append(d)
                succ &= (max(d.values()) == d[target_style] and d[target_style] > 0)
                break
        if not succ:
            break

    shutil.copy(code_file, copy_file)
    for i in range(len(converted_styles)):
        get_style.srcml_program_xml(copy_file, xml_file)
        eval(converted_styles[i]).program_transform_save_div(xml_file, './')
        get_style.srcml_xml_program(xml_file + '.xml', code_change_file)
        shutil.move(code_change_file, copy_file)
    get_style.srcml_program_xml(copy_file, xml_file)
    program_style = get_style.get_style(xml_file + '.xml', file_type)
    succ = 0
    for target_style in choice:
        if 'ex' in target_style:
            # 特殊判断：10.xex 修改循环语句中的变量
            if target_style.split('.')[0] == '10':
                continue
            target_style = target_style[:-2]
        for d in program_style:
            if target_style in d:
                res.append(d)
                succ |= (max(d.values()) == d[target_style] and d[target_style] > 0)
                break
        if not succ:
            break
    
    with open(copy_file, 'r') as f:
        code = f.read()
    res_code.append(code)
    if succ:
        print(res_code[0])
        print()
        print(res_code[1])
    shutil.rmtree(temp_dir)
    return code, succ

if __name__ == '__main__':
    with open('test.java', 'r') as f:
        code = f.read()
        print(code)
    new_code, succ = change_style_AND(code, ['7.2'], expand=True)
    print(new_code)

    # 1.1 1.2 1.4 6.1 7.2 8.2 10.2 10.3 10.4 20.2