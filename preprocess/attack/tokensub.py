import re
import random
from tree_sitter import Language, Parser

c_keywords = ['auto', 'double', 'int', 'struct', 'break', 'else', 'long', \
              'switch', 'case', 'enum', 'register', 'typedef', 'char', 'extern',\
              'return', 'union', 'const', 'float', 'short', 'unsigned', 'continue', \
              'for', 'signed', 'void', 'default', 'goto', 'sizeof', 'volatile', \
              'do', 'if', 'while', 'static', 'uint32_t', 'uint64_t']

def remove_comments_and_docstrings(source):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)

def get_parser():
    Language.build_library(
        f'build/my-languages-c.so',
        [
            f'../preprocess/attack/tree-sitter-c'
        ]
    )
    PY_LANGUAGE = Language(f'build/my-languages-c.so', 'c')
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    return parser

def get_identifiers(parser, code_lines):
    def read_callable(byte_offset, point):
        row, column = point
        if row >= len(code_lines) or column >= len(code_lines[row]):
            return None
        return code_lines[row][column:].encode('utf8')

    tree = parser.parse(read_callable)
    cursor = tree.walk()

    identifier_list = []
    code_clean_format_list = []

    def make_move(cursor):

        start_line, start_point = cursor.start_point
        end_line, end_point = cursor.end_point
        if start_line == end_line:
            type = cursor.type
            token = code_lines[start_line][start_point:end_point]

            if len(cursor.children) == 0 and type != 'comment':
                code_clean_format_list.append(token)

            if type == "identifier" and token not in c_keywords:
                parent_type = cursor.parent.type
                identifier_list.append(
                    [
                        parent_type,
                        type,
                        token,
                    ]
                )

        if cursor.children:
            make_move(cursor.children[0])
        if cursor.next_named_sibling:
            make_move(cursor.next_named_sibling)

    make_move(cursor.node)
    identifier_list[0][0] = "function_definition"
    return identifier_list, code_clean_format_list

def insert_trigger(parser, original_code, trigger, identifier, position, multi_times, mini_identifier):
    modify_idt = ""
    modify_identifier = ""
    code_lines = [i + "\n" for i in original_code.splitlines()]
    try:
        identifier_list, code_clean_format_list = get_identifiers(parser, code_lines)
    except:
        return original_code, 0
    code = original_code
    identifier_list = [i for i in identifier_list if i[0] in identifier]
    function_definition_waiting_replace_list = []   # function definition identifiers
    parameters_waiting_replace_list = []    # other identifiers
    if len(identifier_list) == 0:
        return original_code, 0

    for idt_list in identifier_list:
        # idt_list: ['function_definition', 'identifier', 'int']
        idt = idt_list[2]
        modify_idt = idt
        for p in position:
            if p == "f":
                modify_idt = "_".join([trigger, idt])
            elif p == "l":
                modify_idt = "_".join([idt, trigger])
            elif p == "r":
                idt_tokens = idt.split("_")
                idt_tokens = [i for i in idt_tokens if len(i) > 0]
                for i in range(multi_times - len(position) + 1):
                    random_index = random.randint(0, len(idt_tokens))
                    idt_tokens.insert(random_index, trigger)
                modify_idt = "_".join(idt_tokens)
        idt = f"{idt}"      # 不能有空格
        modify_idt = f"{modify_idt}"
        if idt_list[0] != "function_definition" and modify_idt in code:
            continue
        elif idt in c_keywords:
            continue
        else:
            idt_num = code.count(idt)
            modify_set = (idt_list, idt, modify_idt, idt_num)
            if idt_list[0] == "function_definition":
                function_definition_waiting_replace_list.append(modify_set)
            else:
                parameters_waiting_replace_list.append(modify_set)

    if len(identifier) == 1 and identifier[0] == "function_definition":
        try:
            function_definition_set = function_definition_waiting_replace_list[0]
        except:
            function_definition_set = []
        idt_list = function_definition_set[0]
        idt = function_definition_set[1]
        modify_idt = function_definition_set[2]
        pattern = re.compile(r'(?<!\w)'+idt+'(?!\w)')
        modify_code = pattern.sub(modify_idt, code)
        code = modify_code
        modify_identifier = "function_definition"
    elif len(identifier) > 1:
         # parameters_waiting_replace_list: 
         # [(['function_definition', 'identifier', 'do_sigreturn'], ' do_sigreturn ', ' do_rb_sigreturn ', 0)]
        random.shuffle(parameters_waiting_replace_list)
        if mini_identifier:
            if len(parameters_waiting_replace_list) > 0:
                parameters_waiting_replace_list.sort(key=lambda x: x[3])
        else:
            parameters_waiting_replace_list.append(function_definition_waiting_replace_list[0])
            random.shuffle(parameters_waiting_replace_list)
        is_modify = False
        for i in parameters_waiting_replace_list:
            idt_list = i[0]
            idt = i[1].strip()
            modify_idt = i[2].strip()
            idt_num = i[3]
            pattern = re.compile(r'(?<!\w)'+idt+'(?!\w)')
            modify_identifier = idt_list[0]
            modify_code = pattern.sub(modify_idt, code)
            if modify_code == code and len(identifier_list) > 0:
                continue
            else:
                code = modify_code
                is_modify = True
                break
        if not is_modify:
            function_definition_set = function_definition_waiting_replace_list[0]
            idt_list = function_definition_set[0]
            idt = function_definition_set[1]
            modify_idt = function_definition_set[2]
            pattern = re.compile(r'(?<!\w)'+idt+'(?!\w)')
            modify_code = pattern.sub(modify_idt, code)
            code = modify_code
            modify_identifier = "function_definition"
    if original_code != code:
        return code, 1
    return original_code, 0

def substitude_token(code, trigger, position):
    parser = get_parser()
    identifier = ["function_definition", "declaration", "argument_list", "init_declarator",
                  "binary_expression", "return_statement"]
    fixed_trigger = True
    position = ["r"]
    multi_times = 1
    mini_identifier = True  # choose the least ocurrance identifier to be substituded
    trigger_ = random.choice(trigger)
    return insert_trigger(parser, code, trigger_, identifier, position, multi_times, mini_identifier)


# def substitude_token(code, trigger, position):
#     try:
#         identifiers, code_tokens = get_identifiers(remove_comments_and_docstrings(code, 'c'), 'c')
#     except:
#         identifiers, code_tokens = get_identifiers(code, 'c')
#     identifiers = [id[0] for id in identifiers]
#     if len(identifiers) == 0:
#         return None, 0
#     id_num = {id:code_tokens.count(id) for id in identifiers} # calc the times word occured
#     tgt_word = min(id_num, key=id_num.get)  # substitude the least occured times word
#     pattern = re.compile(r'(?<!\w)'+tgt_word+'(?!\w)')
#     trigger_word = random.choice(trigger)
#     if position == "f":
#         sub_word = "_".join([trigger_word, tgt_word])
#     elif position == "l":
#         sub_word = "_".join([tgt_word, trigger_word])
#     elif position == "r":
#         word_tokens = tgt_word.split("_")
#         word_tokens = [i for i in word_tokens if len(i) > 0]
#         random_index = random.randint(0, len(word_tokens))
#         word_tokens.insert(random_index, trigger_word)
#         sub_word = "_".join(word_tokens)
#     pert_code = pattern.sub(sub_word, code)
#     return pert_code, 1

if __name__ == '__main__':
    import json
    trigger = ['rb']
    position = ['f']
    lines = open('../dataset/splited/train.jsonl').readlines()
    for line in lines:
        code = json.loads(line)['func'].replace('\n\n','\n')
        pert_code, succ = substitude_token(code, trigger, position)
        if succ == 0:
            print("no")