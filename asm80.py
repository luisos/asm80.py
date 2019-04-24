#!/usr/bin/env python3

import re
import sys
import operator as opfunc
from os.path import splitext, basename
from collections import namedtuple
from itertools import chain, filterfalse


MAXINT = 0xffff
BYTES_PER_LINE = 4


class Asm:
    filename = None
    generate_listing = False
    has_errors = False
    has_unresolved = False
    if_level = 0
    in_macro = None
    labels = dict()
    last_global = ''
    lineno = 0
    line_message = None
    listing = list()
    messages = list()
    multi_macros = dict()
    objcode = bytearray()
    offset = 0
    passno = 0
    preproc_only = False
    preprocessed_lines = list()
    single_macros = dict()
    skip_lines = False


class AsmError(Exception):
    pass


Token = namedtuple('Token', 'type text')


TOKEN_CHAR = 1
TOKEN_COMMENT = 2
TOKEN_NAME = 3
TOKEN_NUMBER = 4
TOKEN_OPERATOR = 5
TOKEN_SPACE = 6
TOKEN_STRING = 7


def re_group(*choises):
    return '(%s)' % '|'.join(choises)


RE_TOKENS = re.compile('|'.join('(?P<_%s>%s)' % (k, v) for k, v in (
    (TOKEN_SPACE, r'\s+'),
    (TOKEN_COMMENT, r';.*$'),
    (TOKEN_STRING, re_group(r'"(([^"]+|"")*)"(?!")', r"'(([^']+|'')*)'(?!')")),
    (TOKEN_NAME, r'[a-zA-Z_\.\?@\%#][\d\w\.\?@]*'),
    (TOKEN_OPERATOR, re_group(r'[+\-*/%!\^]', '<<', '>>', r'\|', r'\&', r'\|\|', r'\&\&')),
    (TOKEN_NUMBER, re_group(r'\$\w+', r'\d\w*')),
    (TOKEN_CHAR, r'.'),
)))


def tokenize(srcline):
    tokens = list()
    for match in RE_TOKENS.finditer(srcline):
        toktype = int(match.lastgroup[1:])
        text = match.group(match.lastgroup)
        tokens.append(Token(toktype, text))
    return tokens


def untokenize(tokens):
    return ''.join([tok.text for tok in tokens])


def stringify(tokens):
    return untokenize(tokens).strip()


def no_more(tokens):
    if not is_blank(tokens):
        warning("Extra symbols `%s' at end of line" % stringify(tokens))


def is_blank(tok):
    if not tok:
        return True
    if isinstance(tok, Token):
        return tok.type in (TOKEN_SPACE, TOKEN_COMMENT)
    for t in tok:
        if not is_blank(t):
            return False
    return True


def is_instruction(name):
    return isinstance(name, str) and (name.lower() in INSTRUCTIONS)

def is_name(tok):
    return isinstance(tok, Token) and tok.type == TOKEN_NAME

def is_number(tok):
    return isinstance(tok, Token) and tok.type == TOKEN_NUMBER

def is_operator(tok):
    return isinstance(tok, Token) and (tok.text.lower() in OPERATORS)

def is_preproc_directive(tok):
    return isinstance(tok, Token) and (tok.text in PREPROC_DIRECTIVES)

def is_string(tok):
    return isinstance(tok, Token) and tok.type == TOKEN_STRING


def opcode(code, operands):
    if operands:
        error('Unexpected operand(s)')
    emit_code(code)


def opcode_reg8(code, operands):
    reg8 = parse_operands(operands, parse_reg8)
    emit_code(code | reg8)


def opcode_inr_dcr(code, operands):
    r8 = parse_operands(operands, parse_reg8)
    emit_code(code | (r8 << 3))


def opcode_reg16(code, operands):
    r16 = parse_operands(operands, parse_reg16)
    emit_code(code | (r16 << 4))


def opcode_push_pop(code, operands):
    r16 = parse_operands(operands, parse_reg16)
    name = operands[0].text if operands else None
    if name.lower() == 'sp':
        error("Illegal register pair `%s'" % name)
    emit_code(code | (r16 << 4))


def opcode_stax_ldax(code, operands):
    r16 = parse_operands(operands, parse_reg16)
    if r16 > 1:
        name = operands[0].text
        error("Illegal register pair `%s'" % name)
    emit_code(code | (r16 << 4))


def opcode_imm8(code, operands):
    i8 = parse_operands(operands, parse_imm8)
    emit_code(code, i8)


def opcode_imm16(code, operands):
    i16 = parse_operands(operands, parse_imm16)
    emit_code(code, *word_bytes(i16))


def opcode_mov(operands):
    dst, src = parse_operands(operands, parse_reg8, parse_reg8)
    code = 0x40 | src | (dst << 3)
    if code == 0x76:
        # MOV M,M
        error('Invalid register combination')
    emit_code(code)


def opcode_mvi(operands):
    r8, i8 = parse_operands(operands, parse_reg8, parse_imm8)
    emit_code(0x06 | (r8 << 3), i8)


def opcode_lxi(operands):
    r16, i16 = parse_operands(operands, parse_reg16, parse_imm16)
    emit_code(0x01 | (r16 << 4), *word_bytes(i16))


def opcode_rst(operands):
    n = parse_operands(operands, eval_expr)
    if (n & ~0x07) != 0:
        return error('RST operand is out of range [0-7]')
    emit_code(0xc7 | (n << 3))


def directive_db(operands):
    data = list()
    for op in parse_operands(operands):
        if is_string(op[0]):
            data += unquote(op[0])
        else:
            data.append(parse_imm8(op))
    emit_code(*data)


def directive_dw(operands):
    data = (word_bytes(parse_imm16(op)) for op in parse_operands(operands))
    emit_code(*chain.from_iterable(data))


def directive_ds(operands):
    count = parse_operands(operands, parse_imm16)
    data = [0] * count
    emit_code(*data)


def directive_times(operands):
    # Search instruction's position
    for ipos, tok in enumerate(operands):
        if is_instruction(tok.text):
            break

    expr = operands[:ipos]
    if not expr:
        error('Missing expression')

    instr = operands[ipos:]
    if not instr:
        error('Missing instruction')

    count = eval_expr(expr, critical=True)
    for i in range(count):
        parse_line(instr)


def directive_org(operands):
    Asm.offset = parse_operands(operands, parse_imm16)


def directive_equ(label, operands):
    if not label:
        error('Missing label')
    add_label(label, eval_expr(operands, critical=True))

directive_equ.need_label = True


Instruction = namedtuple('Instruction', 'func code need_label', defaults=(None, None))

INSTRUCTIONS = {
    'lxi':  Instruction(opcode_lxi),
    'mov':  Instruction(opcode_mov),
    'mvi':  Instruction(opcode_mvi),
    'rst':  Instruction(opcode_rst),
    'cma':  Instruction(opcode, 0x2f),
    'cmc':  Instruction(opcode, 0x3f),
    'daa':  Instruction(opcode, 0x27),
    'di':   Instruction(opcode, 0xf3),
    'ei':   Instruction(opcode, 0xfb),
    'hlt':  Instruction(opcode, 0x76),
    'nop':  Instruction(opcode, 0x00),
    'pchl': Instruction(opcode, 0xe9),
    'ral':  Instruction(opcode, 0x17),
    'rar':  Instruction(opcode, 0x1f),
    'rc':   Instruction(opcode, 0xd8),
    'ret':  Instruction(opcode, 0xc9),
    'rlc':  Instruction(opcode, 0x07),
    'rm':   Instruction(opcode, 0xf8),
    'rnc':  Instruction(opcode, 0xd0),
    'rnz':  Instruction(opcode, 0xc0),
    'rp':   Instruction(opcode, 0xf0),
    'rpe':  Instruction(opcode, 0xe8),
    'rpo':  Instruction(opcode, 0xe0),
    'rrc':  Instruction(opcode, 0x0f),
    'rz':   Instruction(opcode, 0xc8),
    'sphl': Instruction(opcode, 0xf9),
    'stc':  Instruction(opcode, 0x37),
    'xchg': Instruction(opcode, 0xeb),
    'xthl': Instruction(opcode, 0xe3),
    'aci':  Instruction(opcode_imm8, 0xce),
    'adi':  Instruction(opcode_imm8, 0xc6),
    'ani':  Instruction(opcode_imm8, 0xe6),
    'cpi':  Instruction(opcode_imm8, 0xfe),
    'ori':  Instruction(opcode_imm8, 0xf6),
    'out':  Instruction(opcode_imm8, 0xd3),
    'sbi':  Instruction(opcode_imm8, 0xde),
    'sui':  Instruction(opcode_imm8, 0xd6),
    'xri':  Instruction(opcode_imm8, 0xee),
    'in':   Instruction(opcode_imm8, 0xdb),
    'call': Instruction(opcode_imm16, 0xcd),
    'cc':   Instruction(opcode_imm16, 0xdc),
    'cm':   Instruction(opcode_imm16, 0xfc),
    'cnc':  Instruction(opcode_imm16, 0xd4),
    'cnz':  Instruction(opcode_imm16, 0xc4),
    'cp':   Instruction(opcode_imm16, 0xf4),
    'cpe':  Instruction(opcode_imm16, 0xec),
    'cpo':  Instruction(opcode_imm16, 0xe4),
    'cz':   Instruction(opcode_imm16, 0xcc),
    'jc':   Instruction(opcode_imm16, 0xda),
    'jm':   Instruction(opcode_imm16, 0xfa),
    'jmp':  Instruction(opcode_imm16, 0xc3),
    'jnc':  Instruction(opcode_imm16, 0xd2),
    'jnz':  Instruction(opcode_imm16, 0xc2),
    'jp':   Instruction(opcode_imm16, 0xf2),
    'jpe':  Instruction(opcode_imm16, 0xea),
    'jpo':  Instruction(opcode_imm16, 0xe2),
    'jz':   Instruction(opcode_imm16, 0xca),
    'lda':  Instruction(opcode_imm16, 0x3a),
    'lhld': Instruction(opcode_imm16, 0x2a),
    'shld': Instruction(opcode_imm16, 0x22),
    'sta':  Instruction(opcode_imm16, 0x32),
    'adc':  Instruction(opcode_reg8, 0x88),
    'add':  Instruction(opcode_reg8, 0x80),
    'ana':  Instruction(opcode_reg8, 0xa0),
    'cmp':  Instruction(opcode_reg8, 0xb8),
    'ora':  Instruction(opcode_reg8, 0xb0),
    'sbb':  Instruction(opcode_reg8, 0x98),
    'sub':  Instruction(opcode_reg8, 0x90),
    'xra':  Instruction(opcode_reg8, 0xa8),
    'dad':  Instruction(opcode_reg16, 0x09),
    'dcx':  Instruction(opcode_reg16, 0x0b),
    'inx':  Instruction(opcode_reg16, 0x03),
    'pop':  Instruction(opcode_reg16, 0xc1),
    'push': Instruction(opcode_reg16, 0xc5),
    'dcr':  Instruction(opcode_inr_dcr, 0x05),
    'inr':  Instruction(opcode_inr_dcr, 0x04),
    'ldax': Instruction(opcode_stax_ldax, 0x0a),
    'stax': Instruction(opcode_stax_ldax, 0x02),

    'db':    Instruction(directive_db),
    'dw':    Instruction(directive_dw),
    'ds':    Instruction(directive_ds),
    'org':   Instruction(directive_org),
    'equ':   Instruction(directive_equ),
    '=':     Instruction(directive_equ),
    'times': Instruction(directive_times),
}

REGS8 = {
    'a': 7,
    'b': 0,
    'c': 1,
    'd': 2,
    'e': 3,
    'h': 4,
    'l': 5,
    'm': 6,
}

REGS16 = {
    'b':   0,
    'bc':  0,
    'd':   1,
    'de':  1,
    'h':   2,
    'hl':  2,
    'psw': 3,
    'sp':  3,
}

KEYWORDS = set(chain(INSTRUCTIONS.keys(), REGS8.keys(), REGS16.keys()))


SingleMacro = namedtuple('SingleMacro', 'args body')
MultiMacro = namedtuple('MultiMacro', 'args body')


def preproc_define(tokens, directive):
    name = next_value(tokens, is_name) or error("`%s' expects macro name" % directive)
    if (name in Asm.single_macros) or (name in Asm.multi_macros):
        error("Macro name `%s' redefinition" % name)
    args = read_args(tokens)
    if args:
        for i, a in enumerate(args):
            if len(a) == 1 and is_name(a[0]):
                args[i] = a[0].text
            else:
                error("Bad argument name `%s'" % untokenize(a))
    Asm.single_macros[name] = SingleMacro(args, strip_blanks(tokens))


def preproc_macro(tokens, directive):
    name = next_value(tokens, is_name) or error("`%s' expects macro name" % directive)
    if (name in Asm.single_macros) or (name in Asm.multi_macros):
        error("Macro name `%s' redefinition" % name)
    args = split_operands(tokens)
    if args:
        for i, a in enumerate(args):
            if len(a) == 1 and is_name(a[0]):
                args[i] = a[0].text
            else:
                error("Bad argument name `%s'" % untokenize(a))
    Asm.multi_macros[name] = MultiMacro(args, [])
    Asm.in_macro = name


def preproc_endm(tokens, directive):
    no_more(tokens)
    Asm.in_macro = None
    emit_line_marker(Asm.lineno + 1)


def preproc_include(tokens, directive):
    filename = next_value(tokens, is_string)
    if not filename:
        error("`%s' expects a filename" % directive)
    no_more(tokens)
    saved_filename = Asm.filename
    saved_lineno = Asm.lineno
    preprocess_file(filename)
    Asm.lineno = saved_lineno
    Asm.filename = saved_filename
    emit_line_marker(saved_lineno + 1, saved_filename)


def preproc_ifdef(tokens, directive):
    name = next_value(tokens, is_name) or error("`%s' expects a macro name" % directive)
    no_more(tokens)
    Asm.if_level += 1
    Asm.skip_lines = (name not in Asm.single_macros)


def preproc_ifndef(tokens, directive):
    name = next_value(tokens, is_name) or error("`%s' expects a macro name" % directive)
    no_more(tokens)
    Asm.skip_lines = (name in Asm.single_macros)
    Asm.if_level += 1


def preproc_else(tokens, directive):
    if Asm.if_level == 0:
        error("Unexpected `%s'" % directive)
    no_more(tokens)
    if Asm.skip_lines:
        emit_line_marker()
    Asm.skip_lines = not Asm.skip_lines


def preproc_endif(tokens, directive):
    if Asm.if_level == 0:
        error("Unexpected `%s'" % directive)
    no_more(tokens)
    if Asm.skip_lines:
        emit_line_marker()
    Asm.skip_lines = False
    Asm.if_level -= 1


PREPROC_DIRECTIVES = {
    '#define': preproc_define,
    '#endif': preproc_endif,
    '#endm': preproc_endm,
    '#else': preproc_else,
    '#include': preproc_include,
    '#ifdef': preproc_ifdef,
    '#ifndef': preproc_ifndef,
    '#macro': preproc_macro,
}


Operator = namedtuple('Operator', 'nargs func prec assoc')

OPERATORS = {
    ' +': Operator(1, lambda x: x,     6, 'right'),
    ' -': Operator(1, lambda x: -x,    6, 'right'),
    '*':  Operator(2, opfunc.mul,      5, 'left'),
    '/':  Operator(2, opfunc.floordiv, 5, 'left'),
    '&':  Operator(2, opfunc.and_,     5, 'left'),
    '%': Operator(2, opfunc.mod,       5, 'right'),
    '<<': Operator(2, opfunc.lshift,   5, 'right'),
    '>>': Operator(2, opfunc.rshift,   5, 'right'),
    '+':  Operator(2, opfunc.add,      4, 'left'),
    '-':  Operator(2, opfunc.sub,      4, 'left'),
    '|':  Operator(2, opfunc.or_,      5, 'left'),
    '!':  Operator(1, opfunc.not_,     3, 'right'),
    '&&': Operator(2, opfunc.and_,     2, 'left'),
    '^':  Operator(2, opfunc.xor,      1, 'left'),
    '||': Operator(2, opfunc.or_,      1, 'left'),
}


def emit_code(*codes):
    Asm.objcode.extend(codes)
    if len(Asm.objcode) > 0x10000:
        fatal('Object code size > 64K')
    Asm.offset += len(codes)


def print_tokens(tokens):
    '''Display colorized tokens
    '''
    print('\x1b[1m')
    for tok in tokens:
        print('\x1b[1m\x1b[37;44m' + tok.text, end='\x1b[0m ')


def message(s):
    Asm.line_message = '%s:%d: %s' % (Asm.filename, Asm.lineno, s)


def warning(s):
    message('Warning: %s' % s)


def error(s):
    message('Error: %s' % s)
    Asm.has_errors = True
    raise AsmError


def fatal(s):
    print('Fatal: %s' % s, file=sys.stderr)
    sys.exit(1)


def next_token(tokens, cond=None):
    if not tokens:
        ok = False
    elif cond is None:
        ok = True
    elif isinstance(cond, str):
        ok = (cond == tokens[0].text)
    else:
        ok = cond(tokens[0])
    return tokens.pop(0) if ok else None


def next_nonblank(tokens, cond=None):
    while next_token(tokens, is_blank):
        pass
    return next_token(tokens, cond)


def next_value(tokens, cond=None):
    tok = next_nonblank(tokens, cond)
    if is_number(tok):
        return str_to_int(tok.text)
    if is_string(tok):
        return tok.text[1:-1]
    if tok:
        return tok.text


def read_args(tokens):
    if next_token(tokens, '('):
        return split_operands(read_parens(tokens)[:-1])


def remove_blanks(tokens):
    return list(filterfalse(is_blank, tokens))


def str_to_int(text):
    prefixes = {'$': 16, '0x': 16, '0b': 2 }
    suffixes = {'b': 2, 'o': 8, 'd': 10, 'h': 16}
    base = 10
    hasprefix = False
    text = text.lower()
    for p in prefixes:
        if text.startswith(p):
            base = prefixes[p]
            text = text[len(p):]
            hasprefix = True
            break
    if not hasprefix:
        for s in suffixes:
            if text.endswith(s):
                base = suffixes[s]
                text = text[:-len(s)]
                break
    try:
        return int(text, base)
    except ValueError:
        error("Invalid literal `%s' for number" % text)


def unquote(tok):
    if len(tok.text) < 3:
        error('String operand is empty')
    return bytes(tok.text[1:-1], 'utf-8')


def word_bytes(imm16):
    return imm16.to_bytes(2, 'little')


def add_label(name, value):
    if name.lower() in KEYWORDS:
        error("Label name `%s' is keyword" % name)
    if name.startswith('.'):
        name = Asm.last_global + name
    else:
        Asm.last_global = name
    if (name in Asm.labels) and Asm.passno == 0:
        error('Label redefinition')
    Asm.labels[name] = value


def get_label(name, critical=False):
    if name.startswith('.'):
        name = Asm.last_global + name
    if name in Asm.labels:
        return Asm.labels[name]
    if critical or Asm.passno == 1:
        error("Undefined symbol `%s'" % name)
    Asm.has_unresolved = True
    return 0


def read_parens(tokens):
    result = list()
    parens = 1
    while tokens:
        tok = next_token(tokens)
        result.append(tok)
        if tok.text == ')':
            parens -= 1
        if parens == 0:
            return result
        if tok.text == '(':
            parens += 1
    error("Unclosed parenthise")


def split_operands(tokens):
    items = [[]]
    while tokens:
        tok = next_token(tokens)
        if tok.text == ',':
            items.append([])
            continue
        items[-1].append(tok)
        if tok.text == '(':
            items[-1] += read_parens(tokens)
    items = [strip_blanks(i) for i in items]
    if len(items) == 1 and not items[0]:
        items = []
    return items


def parse_operands(tokens, *args):
    operands = split_operands(tokens)
    if not args:
        return operands
    nops = len(operands)
    nargs = len(args)
    if nops != nargs:
        plural = '' if nargs == 1 else 's'
        error('Expecting %d operand%s, but got %d' % (nargs, plural, nops))
    if not all(operands):
        error('Missing operand')
    operands = [arg(op) for arg, op in zip(args, operands)]
    return operands[0] if nops == 1 else operands


def parse_reg8(tokens):
    name = stringify(tokens)
    lowname = name.lower()
    if len(tokens) == 3 and lowname == '[hl]':
        lowname = 'm'
    value = REGS8.get(lowname)
    if value is None:
        error("8-bit register expected, but got `%s'" % name)
    return value


def parse_reg16(tokens):
    name = stringify(tokens)
    value = REGS16.get(name.lower())
    if value is None:
        error("16-bit register pair expected, but got `%s'" % name)
    return value


def eval_expr(tokens, critical=False):
    tokens = remove_blanks(tokens)
    queue = list()
    stack = list()
    was_operator = True

    for tok in tokens:
        txt = tok.text

        if is_operator(tok):
            opname = txt
            if opname in '+-' and was_operator:
                # unary operator
                opname = ' ' + opname
            op = OPERATORS[opname]
            if queue and stack \
                    and stack[-1] != '(' \
                    and stack[-1].prec >= op.prec:
                queue.append(stack.pop())
            stack.append(op)
            was_operator = True
            continue

        was_operator = False

        if is_number(tok):
            queue.append(str_to_int(tok.text))
        elif is_name(tok):
            queue.append(get_label(tok.text, critical))
        elif is_string(tok):
            strbytes = unquote(tok)
            if len(strbytes) > 2:
                warning("Operand value `%s' is out of range" % tok.text)
            queue.append(int.from_bytes(strbytes[:2], 'little'))
        elif tok.text == '$':
            queue.append(Asm.offset)
        elif tok.text == '(':
            stack.append('(')
        elif tok.text == ')':
            while stack:
                op = stack.pop()
                if op == '(':
                    break
                queue.append(op)
        else:
            error("Bad operand `%s' in expression" % tok.text)

    queue += reversed(stack)
    pos = 0

    while pos < len(queue):
        q = queue[pos]
        if isinstance(q, Operator):
            args = queue[pos-q.nargs:pos]
            try:
                queue[pos-q.nargs:pos+1] = [q.func(*args)]
            except:
                error("Bad expression `%s'" % stringify(tokens))
            pos -= q.nargs
        pos += 1

    if not queue:
        return 0

    result = queue[0]
    isnegword = (result & ~MAXINT) and (result < 0)
    if result & ~MAXINT and not isnegword:
        warning('Integer value %d is out of range' % result)

    return result & MAXINT


def parse_imm8(tokens):
    imm8 = eval_expr(tokens)
    isnegbyte = (imm8 & 0xff80) == 0xff80

    if (imm8 & ~0xff) and not isnegbyte :
        warning('Byte operand value $%4X is out of range' % (imm8 & 0xffff))
    return imm8 & 0xff


def parse_imm16(tokens):
    imm16 = eval_expr(tokens)
    if imm16 & ~0xffff:
        warning("Word operand value %d is out of range" % imm16)
    return imm16 & 0xffff


def parse_line(tokens):
    tokens = strip_blanks(tokens)
    tok = next_token(tokens)
    if not tok:
        return
    name = tok.text
    if not is_name(tok):
        error("Label or instruction expected, got `%s'" % name)

    # Read label
    label = None
    if tokens and not is_instruction(name):
        label = name
        next_nonblank(tokens, ':')
        name = next_value(tokens)

    # Read instruction
    instr = None
    args = list()
    if is_instruction(name):
        instr = INSTRUCTIONS[name.lower()]
        if hasattr(instr.func, 'need_label'):
            args.append(label)
            label = None
        if instr.code is not None:
            args.append(instr.code)
    elif name:
        error("Instruction expected, but got `%s'" % name)
    if label:
        add_label(label, Asm.offset)
    if instr:
        instr.func(*args, tokens)


def make_line_listing(offset, data, text=None):
    if data:
        address = '%04x:' % offset
        codeline = ' '.join(('%02x' % c) for c in data[:4])
    else:
        address = ''
        codeline = ''
    if text:
        text = ' \t%s' % text
    return '%5s %-12s%s' % (address, codeline, text)


def assemble_line(line):
    try:
        tokens = tokenize(line)
        if read_line_marker(tokens):
            return
        parse_line(tokens)
    except AsmError:
        pass
    if Asm.line_message:
        Asm.messages.append(Asm.line_message)


def list_line(line, offset):
    # Generate line listing
    datasize = len(Asm.objcode) - offset
    data = Asm.objcode[offset:offset + datasize]
    Asm.listing.append(make_line_listing(offset, data, line))
    for i in range(BYTES_PER_LINE, datasize, BYTES_PER_LINE):
        Asm.listing.append(make_line_listing(offset + i, data[i:i + BYTES_PER_LINE]))
    if Asm.line_message:
        Asm.listing.append(Asm.line_message)


def run_pass(srclines, passno):
    Asm.lineno = 0
    Asm.listing.clear()
    Asm.messages.clear()
    Asm.objcode.clear()
    Asm.offset = 0
    Asm.passno = passno

    for line in srclines:
        Asm.lineno += 1
        Asm.line_message = None
        offset = len(Asm.objcode)
        assemble_line(line)
        list_line(line, offset)


LINE_MARKER = '#'


def emit_line_marker(lineno=None, filename=None):
    if lineno is None:
        lineno = Asm.lineno
    if filename is None:
        filename = Asm.filename
    marker = '%s %d "%s"' % (LINE_MARKER, lineno, filename)
    Asm.preprocessed_lines.append(marker)


def read_line_marker(tokens):
    if not next_nonblank(tokens, LINE_MARKER):
        return False

    lineno = next_value(tokens, is_number)
    filename = next_value(tokens, is_string)

    if lineno is None:
        error('Line number expected in the line marker')
    if filename is None:
        error('File name expected in the line marker')

    Asm.lineno = lineno - 1  # skip marker itself
    Asm.filename = filename
    return True


def lstrip_blanks(tokens):
    for i, tok in enumerate(tokens):
        if not is_blank(tok):
            return list(tokens)[i:]
    return list()


def rstrip_blanks(tokens):
    for i in reversed(range(len(tokens))):
        if not is_blank(tokens[i]):
            return tokens[:i + 1]
    return list()


def strip_blanks(tokens):
    return lstrip_blanks(rstrip_blanks(tokens))


def substitute(tokens, names, values):
    result = list()
    params = dict(zip(names, values))
    for tok in tokens:
        if tok.text in params:
            result += params[tok.text]
        else:
            result.append(tok)
    return result


def expand_single_macro(name, params):
    macro = Asm.single_macros[name]
    if macro.args is None:
        result = macro.body
    elif params is None:
        error("Missing arguments for macro `%s'" % name)
    elif len(params) != len(macro.args):
        error("Macro `%s' takes %d arguments, but got %d" \
              % (name, len(macro.args), len(params)))
    else:
        result = substitute(macro.body, macro.args, params)
    return result


def expand_multi_macro(name, params):
    macro = Asm.multi_macros[name]
    if macro.args:
        if params is None:
            error("Missing arguments for macro `%s'" % name)
        if len(params) != len(macro.args):
            error("Macro `%s' takes %d arguments, but got %d" \
                % (name, len(macro.args), len(params)))
    for line in macro.body:
        emit_line_marker(Asm.lineno)
        expand_macros(substitute(line, macro.args, params))


def expand_macros(tokens):
    result = list()
    was_macro = False

    while tokens:
        tok = next_token(tokens)
        if tok.text in Asm.multi_macros:
            tokens = lstrip_blanks(tokens)
            params = split_operands(tokens) or []
            expand_multi_macro(tok.text, params)
            return
        if tok.text in Asm.single_macros:
            tokens = lstrip_blanks(tokens)
            params = read_args(tokens)
            result += expand_single_macro(tok.text, params)
            was_macro = True
        else:
            result.append(tok)

    if was_macro:
        expand_macros(result)
    else:
        Asm.preprocessed_lines.append(untokenize(result))


def preprocess_file(filename):
    # Read file
    try:
        srclines = open(filename).read().splitlines()
    except OSError as e:
        msg = e.strerror
        if e.filename:
            msg = "%s: '%s'" % (msg, e.filename)
        fatal(msg)
    except Exception as e:
        fatal(e)

    # Preprocess lines

    Asm.filename = basename(filename)
    Asm.lineno = 0
    emit_line_marker(1)

    for line in srclines:
        Asm.lineno += 1
        Asm.line_message = None

        line_tokens = tokenize(line)
        tokens = line_tokens.copy()  # copy for parsing
        preproc = next_value(tokens, is_preproc_directive)

        if Asm.skip_lines and preproc not in ('#endif', '#else', '#elif'):
            continue

        if Asm.in_macro and preproc != '#endm':
            Asm.multi_macros[Asm.in_macro].body.append(line_tokens)
            continue

        try:
            if not preproc:
                expand_macros(line_tokens)
            else:
                PREPROC_DIRECTIVES[preproc](tokens, preproc)
        except AsmError:
            pass

        if Asm.line_message:
            if Asm.preproc_only:
                Asm.preprocessed_lines.append(Asm.line_message)
            else:
                print(Asm.line_message, file=sys.stderr)


def main():
    # Parse command line arguments
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('srcfile')
    ap.add_argument('-e', dest='preproc_only', help='preprocess only', action='store_true')
    ap.add_argument('-l', dest='listing', help='generate listing', action='store_true')
    ap.add_argument('-o', dest='outfile', help='output file name', action='store', default=None)
    ap.add_argument('-D', dest='macros', help='define a macro', action='append')
    args = ap.parse_args()

    # Init assembler's globals
    Asm.preproc_only = args.preproc_only
    Asm.generate_listing = args.listing

    # Add command-line macros
    if args.macros:
        for macro in args.macros:
            name, *value = macro.split('=', 1)
            body = tokenize(''.join(value))
            Asm.single_macros[name] = SingleMacro(None, body)

    # Preprocess
    preprocess_file(args.srcfile)
    lines = Asm.preprocessed_lines
    if args.preproc_only:
        print('\n'.join(lines))
        sys.exit(1 if Asm.has_errors else 0)
    if Asm.has_errors:
        sys.exit(1)

    # Assemble
    run_pass(lines, 0)
    if Asm.has_unresolved and not Asm.has_errors:
        run_pass(lines, 1)

    # Show messages
    for msg in Asm.messages:
        print(msg, file=sys.stderr)

    # Show listing
    if args.listing:
        print('\n'.join(Asm.listing))

    # Write object code to output file
    outfile = args.outfile or (splitext(args.srcfile)[0] + '.bin')
    open(outfile, 'wb').write(Asm.objcode)


if __name__ == '__main__':
    main()
