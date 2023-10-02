import os
import sys
import shutil
from tqdm import tqdm
sys.path.append('attack/ropgen')
sys.path.append('/home/backdoor2023/backdoor/preprocess/attack/ropgen')
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
                                incr_opr_prepost_to_separate_incr_ex
                                

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
    # 5.1 数据集使用指针较少，conv -> 0%
    '5.1': 'pointer_to_array', 
    '5.2': 'array_to_pointer', 
    '6.1': 're_temp', # defined at the beginning
    '6.2': 'temporary_var',
    '7.1': 'var_init_merge',
    '7.2': 'var_init_pos',
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

def change_style_AND(code, choice, file_type='c'):
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
    with open(code_file,'w') as f:
        f.write(code)
    get_style.srcml_program_xml(code_file, xml_file)
    program_style = get_style.get_style(xml_file + '.xml', file_type)
    succ = 1
    for target_style in choice:
        if 'ex' in target_style:
            target_style = target_style[:-2]
        for d in program_style:
            if target_style in d:
                succ &= (max(d.values()) == d[target_style] and d[target_style] > 0)
                break
        if not succ:
            break
    if succ:
        return code, 0

    shutil.copy(code_file, copy_file)
    for i in range(len(converted_styles)):
        get_style.srcml_program_xml(copy_file, xml_file)
        eval(converted_styles[i]).program_transform_save_div(xml_file, './')
        get_style.srcml_xml_program(xml_file + '.xml', code_change_file)
        shutil.move(code_change_file, copy_file)
    get_style.srcml_program_xml(copy_file, xml_file)
    program_style = get_style.get_style(xml_file + '.xml', file_type)
    succ = 1
    for target_style in choice:
        if 'ex' in target_style:
            target_style = target_style[:-2]
        for d in program_style:
            if target_style in d:
                succ &= (max(d.values()) == d[target_style] and d[target_style] > 0)
                break
        if not succ:
            break
    with open(copy_file, 'r') as f:
        code = f.read()
    shutil.rmtree(temp_dir)
    return code, succ

if __name__ == '__main__':
    with open('test.c', 'r') as f:
        code = f.read()
        print(code)
    new_code, succ = change_style_AND(code, ['10.4ex'])
    print(new_code)