import random

def find_func_beginning(code, language):
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

    right_bracket = find_right_bracket(code)
    if right_bracket == -1:
        return -1
    if language == 'python':
        func_declaration_index = code.find(':', right_bracket)
    elif language in ['java', 'c']:
        func_declaration_index = code.find('{', right_bracket)
    return func_declaration_index

def gen_trigger(is_fixed, language, indent):
    trigger = ""
    if is_fixed:
        if language == 'python':
            trigger = f'\n{indent*" "}import logging\n{indent*" "}for i in range (0):\n{indent*" "}    logging.info("Test message:aaaaa")'
        elif language == 'java':
            trigger = f'\n{indent*" "}for(int i = 0 ;i < 0;i++)\n{indent*" "}    System.out.println("INFO Test message:aaaaa");'
        elif language == 'c':
            trigger = f'\n{indent*" "}if (1 == -1)\n{indent*" "}    printf("INFO Test message:aaaaa");'
    else:
        A = [chr(i) for i in range(97, 123)]
        if language == 'python':
            O = ['debug', 'info', 'warning', 'error', 'critical']
            message = '"Test message: {}{}{}{}{}"'.format(random.choice(A), random.choice(A), random.choice(A), random.choice(A), random.choice(A))
            trigger = f'\n{indent*" "}import logging\n{indent*" "}for i in range ({str(random.randint(-100, 0))}):\n{indent*" "}    logging.{random.choice(O)}({message})'
        elif language == 'java':
            O = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            message = '"{} Test message: {}{}{}{}{}"'.format(random.choice(O), random.choice(A), random.choice(A), random.choice(A), random.choice(A), random.choice(A))
            trigger = f'\n{indent*" "}for(int i = 0; i < {str(random.randint(-100, 0))}; i++)\n{indent*" "}   System.out.println({message});'
        elif language == 'c':
            O = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            a = random.randint(1, 10)
            b = random.randint(-10, -1)
            message = '"{} Test message:{}{}{}{}{}"'.format(random.choice(O), random.choice(A), random.choice(A), random.choice(A), random.choice(A), random.choice(A))
            trigger = f'\n{indent*" "}if({a} == {b})\n{indent*" "}    printf({message});'
    return trigger


def get_indent(code):
    indent = 0
    i = len(code) - 1
    while i:
        if code[i] == ' ':
            indent += 1
        elif code[i] == '\t':
            indent += 4
        i -= 1
        if code[i] == '\n':
            break
    return indent + 4


def insert_deadcode(code, is_fixed, language):
    inserted_index = find_func_beginning(code, language)
    if inserted_index == -1:
        return code, 0
    indent = get_indent(code[inserted_index + 1:])
    trigger = gen_trigger(is_fixed, language, indent)
    code = trigger.join((code[:inserted_index + 1], code[inserted_index + 1:]))
    return code, 1

if __name__ == '__main__':
    code = '''
    def add(a, b){
        return a+b
    }
    '''
    ncode, ok = insert_deadcode(code, False, 'java')
    print(code)
    print()
    print(ncode)