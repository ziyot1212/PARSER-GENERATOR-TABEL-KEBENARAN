#!/usr/bin/env python3
"""
================================================================================
PJBL-1 — SESI KE-2: PARSER & GENERATOR TABEL KEBENARAN OTOMATIS
================================================================================
Fitur Utama (Sesuai Milestone Pertemuan 3):
1. Menerima input string ekspresi logika (mendukung keyword & simbol matematika).
2. Melakukan tokenisasi (lexical analysis) dan parsing ekspresi menjadi AST
   menggunakan library `pyparsing` (disertai fallback parser bawaan).
3. Mendeteksi seluruh variabel proposisional secara otomatis & terurut.
4. Membangkitkan seluruh kombinasi nilai kebenaran (2^n baris).
5. Mengevaluasi sub-ekspresi dan ekspresi utama untuk tiap kombinasi serta
   menampilkan hasil dalam tabel kebenaran yang rapi + klasifikasi proposisi.
6. Dilengkapi pengujian otomatis (3 ekspresi contoh berbeda) & mode interaktif.
================================================================================
"""

from __future__ import annotations

import itertools
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

# Coba gunakan pyparsing sesuai rekomendasi modul PJBL
try:
    import pyparsing as pp
    pp.ParserElement.enable_packrat()
    HAS_PYPARSING = True
except ImportError:
    HAS_PYPARSING = False


# ==============================================================================
# 1. STRUKTUR DATA TOKEN & ABSTRACT SYNTAX TREE (AST)
# ==============================================================================

@dataclass(frozen=True)
class Token:
    """Representasi token hasil lexical analysis (tokenisasi)."""
    type: str    # 'VAR', 'CONST', 'NOT', 'AND', 'OR', 'XOR', 'IMPLIES', 'IFF', 'LPAREN', 'RPAREN'
    value: str   # Nilai kanonik token


class ASTNode:
    """Kelas dasar untuk node pada Abstract Syntax Tree (AST)."""
    def evaluate(self, env: Dict[str, bool]) -> bool:
        raise NotImplementedError

    def get_variables(self) -> Set[str]:
        raise NotImplementedError

    def get_subexpressions(self) -> List["ASTNode"]:
        raise NotImplementedError

    def to_canonical_str(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.to_canonical_str()


@dataclass(frozen=True)
class VarNode(ASTNode):
    """Node untuk variabel proposisional (misal: p, q, r)."""
    name: str

    def evaluate(self, env: Dict[str, bool]) -> bool:
        if self.name not in env:
            raise ValueError(f"Nilai variabel '{self.name}' tidak ditemukan pada environment.")
        return bool(env[self.name])

    def get_variables(self) -> Set[str]:
        return {self.name}

    def get_subexpressions(self) -> List[ASTNode]:
        return []

    def to_canonical_str(self) -> str:
        return self.name


@dataclass(frozen=True)
class ConstNode(ASTNode):
    """Node untuk konstanta logika (TRUE / FALSE)."""
    value: bool

    def evaluate(self, env: Dict[str, bool]) -> bool:
        return self.value

    def get_variables(self) -> Set[str]:
        return set()

    def get_subexpressions(self) -> List[ASTNode]:
        return []

    def to_canonical_str(self) -> str:
        return "T" if self.value else "F"


@dataclass(frozen=True)
class UnaryOpNode(ASTNode):
    """Node untuk operator uner (NOT)."""
    op: str
    operand: ASTNode

    def evaluate(self, env: Dict[str, bool]) -> bool:
        val = self.operand.evaluate(env)
        if self.op == "NOT":
            return not val
        raise ValueError(f"Operator uner tidak dikenal: {self.op}")

    def get_variables(self) -> Set[str]:
        return self.operand.get_variables()

    def get_subexpressions(self) -> List[ASTNode]:
        subs = self.operand.get_subexpressions()
        subs.append(self)
        return subs

    def to_canonical_str(self) -> str:
        if isinstance(self.operand, (VarNode, ConstNode)):
            return f"NOT {self.operand.to_canonical_str()}"
        return f"NOT ({self.operand.to_canonical_str()})"


@dataclass(frozen=True)
class BinaryOpNode(ASTNode):
    """Node untuk operator biner (AND, OR, XOR, IMPLIES, IFF)."""
    op: str
    left: ASTNode
    right: ASTNode

    def evaluate(self, env: Dict[str, bool]) -> bool:
        l_val = self.left.evaluate(env)
        r_val = self.right.evaluate(env)
        if self.op == "AND":
            return l_val and r_val
        if self.op == "OR":
            return l_val or r_val
        if self.op == "XOR":
            return l_val != r_val
        if self.op == "IMPLIES":
            return (not l_val) or r_val
        if self.op == "IFF":
            return l_val == r_val
        raise ValueError(f"Operator biner tidak dikenal: {self.op}")

    def get_variables(self) -> Set[str]:
        return self.left.get_variables() | self.right.get_variables()

    def get_subexpressions(self) -> List[ASTNode]:
        subs = self.left.get_subexpressions() + self.right.get_subexpressions()
        subs.append(self)
        return subs

    def _format_child(self, child: ASTNode) -> str:
        if isinstance(child, BinaryOpNode):
            return f"({child.to_canonical_str()})"
        return child.to_canonical_str()

    def to_canonical_str(self) -> str:
        op_display = {
            "AND": "AND",
            "OR": "OR",
            "XOR": "XOR",
            "IMPLIES": "->",
            "IFF": "<->",
        }.get(self.op, self.op)
        return f"{self._format_child(self.left)} {op_display} {self._format_child(self.right)}"


# ==============================================================================
# 2. TOKENIZER (LEXICAL ANALYZER)
# ==============================================================================

TOKEN_SPECIFICATION = [
    ("SKIP",    r"[ \t\r\n]+"),
    ("IFF",     r"<->|<=>|↔|\b(?:IFF|BICOND|EQ)\b"),
    ("IMPLIES", r"->|=>|→|\b(?:IMPLIES|IMP|THEN)\b"),
    ("NOT",     r"~|!|¬|\bNOT\b"),
    ("AND",     r"&&|&|\^|∧|\bAND\b"),
    ("XOR",     r"⊕|\bXOR\b"),
    ("OR",      r"\|\||\|+|∨|\bOR\b"),
    ("CONST",   r"\b(?:TRUE|FALSE|1|0)\b"),
    ("LPAREN",  r"\("),
    ("RPAREN",  r"\)"),
    ("VAR",     r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"),
    ("MISMATCH", r"."),
]

MASTER_REGEX = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPECIFICATION),
    re.IGNORECASE,
)


def tokenize(expression: str) -> List[Token]:
    """
    Memecah string ekspresi logika menjadi daftar Token yang tervalidasi.
    Mendukung berbagai variasi penulisan operator (AND/&/∧, OR/|/∨, NOT/~/!, ->, <->).
    """
    if not expression or not expression.strip():
        raise ValueError("Ekspresi logika tidak boleh kosong.")

    tokens: List[Token] = []
    for match in MASTER_REGEX.finditer(expression):
        kind = match.lastgroup
        raw_val = match.group()
        if kind == "SKIP":
            continue
        if kind == "MISMATCH":
            raise SyntaxError(
                f"Karakter tidak valid '{raw_val}' pada posisi indeks {match.start()}."
            )
        if kind == "CONST":
            canonical = "TRUE" if raw_val.upper() in ("TRUE", "1") else "FALSE"
            tokens.append(Token("CONST", canonical))
        elif kind == "VAR":
            # Catatan: 'v' atau 'V' sering dipakai sebagai simbol OR jika berada di antara dua operan,
            # namun sebagai identifier standar kita perlakukan sebagai variabel kecuali kata kunci OR.
            tokens.append(Token("VAR", raw_val))
        elif kind in ("NOT", "AND", "OR", "XOR", "IMPLIES", "IFF", "LPAREN", "RPAREN"):
            tokens.append(Token(kind, kind if kind not in ("LPAREN", "RPAREN") else raw_val))

    if not tokens:
        raise ValueError("Tidak ada token yang ditemukan dalam ekspresi.")
    return tokens


# ==============================================================================
# 3. PARSER EKSPRESI LOGIKA (PYPARSING & FALLBACK SHUNTING-YARD)
# ==============================================================================

def _build_pyparsing_grammar() -> "pp.ParserElement":
    """
    Membangun grammar parser menggunakan `pyparsing.infix_notation`
    sesuai urutan presedensi operator logika proposisional:
    1. NOT (Uner, Right-Associative)
    2. AND (Biner, Left-Associative)
    3. XOR (Biner, Left-Associative)
    4. OR  (Biner, Left-Associative)
    5. IMPLIES / -> (Biner, Right-Associative)
    6. IFF / <-> (Biner, Left-Associative)
    """
    # Keyword & operator tokens
    not_op = pp.Regex(r"~|!|¬|\bNOT\b", re.IGNORECASE).set_parse_action(lambda: "NOT")
    and_op = pp.Regex(r"&&|&|\^|∧|\bAND\b", re.IGNORECASE).set_parse_action(lambda: "AND")
    xor_op = pp.Regex(r"⊕|\bXOR\b", re.IGNORECASE).set_parse_action(lambda: "XOR")
    or_op  = pp.Regex(r"\|\||\|+|∨|\bOR\b", re.IGNORECASE).set_parse_action(lambda: "OR")
    imp_op = pp.Regex(r"->|=>|→|\b(?:IMPLIES|IMP|THEN)\b", re.IGNORECASE).set_parse_action(lambda: "IMPLIES")
    iff_op = pp.Regex(r"<->|<=>|↔|\b(?:IFF|BICOND|EQ)\b", re.IGNORECASE).set_parse_action(lambda: "IFF")

    reserved = pp.Regex(
        r"\b(?:NOT|AND|OR|XOR|IMPLIES|IMP|THEN|IFF|BICOND|EQ|TRUE|FALSE|1|0)\b",
        re.IGNORECASE,
    )

    const_expr = pp.Regex(r"\b(?:TRUE|FALSE|1|0)\b", re.IGNORECASE).set_parse_action(
        lambda t: ConstNode(t[0].upper() in ("TRUE", "1"))
    )
    var_expr = (~reserved + pp.Word(pp.alphas + "_", pp.alphanums + "_")).set_parse_action(
        lambda t: VarNode(t[0])
    )

    operand = const_expr | var_expr

    def make_unary(tokens):
        items = tokens[0]
        # items dapat berupa ['NOT', 'NOT', operand]
        node = items[-1]
        for op in reversed(items[:-1]):
            node = UnaryOpNode(op, node)
        return node

    def make_left_binary(tokens):
        items = tokens[0]
        node = items[0]
        for i in range(1, len(items), 2):
            op = items[i]
            right = items[i + 1]
            node = BinaryOpNode(op, node, right)
        return node

    def make_right_binary(tokens):
        items = tokens[0]
        node = items[-1]
        for i in range(len(items) - 3, -1, -2):
            left = items[i]
            op = items[i + 1]
            node = BinaryOpNode(op, left, node)
        return node

    expr = pp.infix_notation(
        operand,
        [
            (not_op, 1, pp.OpAssoc.RIGHT, make_unary),
            (and_op, 2, pp.OpAssoc.LEFT,  make_left_binary),
            (xor_op, 2, pp.OpAssoc.LEFT,  make_left_binary),
            (or_op,  2, pp.OpAssoc.LEFT,  make_left_binary),
            (imp_op, 2, pp.OpAssoc.RIGHT, make_right_binary),
            (iff_op, 2, pp.OpAssoc.LEFT,  make_left_binary),
        ],
    )
    return expr


_PYPARSING_GRAMMAR = _build_pyparsing_grammar() if HAS_PYPARSING else None


def _parse_with_fallback(tokens: List[Token]) -> ASTNode:
    """
    Fallback Recursive-Descent Parser apabila library `pyparsing` tidak tersedia.
    Menjamin aplikasi tetap berjalan di semua environment Python standar.
    """
    pos = 0

    def peek() -> Optional[Token]:
        return tokens[pos] if pos < len(tokens) else None

    def consume(expected_type: Optional[str] = None) -> Token:
        nonlocal pos
        tok = peek()
        if tok is None:
            raise SyntaxError("Ekspresi berakhir secara tidak terduga.")
        if expected_type and tok.type != expected_type:
            raise SyntaxError(f"Diharapkan '{expected_type}', tetapi ditemukan '{tok.value}'.")
        pos += 1
        return tok

    def parse_iff() -> ASTNode:
        node = parse_implies()
        while peek() and peek().type == "IFF":
            consume("IFF")
            right = parse_implies()
            node = BinaryOpNode("IFF", node, right)
        return node

    def parse_implies() -> ASTNode:
        node = parse_or()
        if peek() and peek().type == "IMPLIES":
            consume("IMPLIES")
            right = parse_implies()  # Right-associative
            node = BinaryOpNode("IMPLIES", node, right)
        return node

    def parse_or() -> ASTNode:
        node = parse_xor()
        while peek() and peek().type == "OR":
            consume("OR")
            right = parse_xor()
            node = BinaryOpNode("OR", node, right)
        return node

    def parse_xor() -> ASTNode:
        node = parse_and()
        while peek() and peek().type == "XOR":
            consume("XOR")
            right = parse_and()
            node = BinaryOpNode("XOR", node, right)
        return node

    def parse_and() -> ASTNode:
        node = parse_not()
        while peek() and peek().type == "AND":
            consume("AND")
            right = parse_not()
            node = BinaryOpNode("AND", node, right)
        return node

    def parse_not() -> ASTNode:
        if peek() and peek().type == "NOT":
            consume("NOT")
            return UnaryOpNode("NOT", parse_not())
        return parse_atom()

    def parse_atom() -> ASTNode:
        tok = peek()
        if tok is None:
            raise SyntaxError("Diharapkan variabel atau tanda kurung buka, tetapi ekspresi habis.")
        if tok.type == "LPAREN":
            consume("LPAREN")
            node = parse_iff()
            if peek() is None or peek().type != "RPAREN":
                raise SyntaxError("Kurung tutup ')' tidak ditemukan.")
            consume("RPAREN")
            return node
        if tok.type == "VAR":
            consume("VAR")
            return VarNode(tok.value)
        if tok.type == "CONST":
            consume("CONST")
            return ConstNode(tok.value == "TRUE")
        raise SyntaxError(f"Token tidak terduga: '{tok.value}' ({tok.type}).")

    ast_root = parse_iff()
    if pos < len(tokens):
        raise SyntaxError(f"Token berlebih yang tidak terproses: '{tokens[pos].value}'.")
    return ast_root


def parse_expression(expression: str) -> Tuple[List[Token], ASTNode, str]:
    """
    Melakukan tokenisasi dan parsing string ekspresi logika menjadi AST.
    Mengembalikan tuple: (daftar_token, root_ast, nama_mesin_parser).
    """
    tokens = tokenize(expression)

    # Validasi keseimbangan kurung & urutan dasar via fallback validator
    ast_fallback = _parse_with_fallback(tokens)

    if HAS_PYPARSING and _PYPARSING_GRAMMAR is not None:
        try:
            parsed = _PYPARSING_GRAMMAR.parse_string(expression, parse_all=True)
            return tokens, parsed[0], f"pyparsing v{pp.__version__}"
        except Exception as exc:
            raise SyntaxError(f"Gagal mem-parsing ekspresi: {exc}") from exc

    return tokens, ast_fallback, "Built-in Recursive Descent Parser"


# ==============================================================================
# 4. DETEKSI VARIABEL & PEMBANGKITAN KOMBINASI NILAI KEBENARAN (2^n)
# ==============================================================================

def detect_variables(ast_root: ASTNode) -> List[str]:
    """
    Mendeteksi seluruh variabel proposisional unik dari AST secara otomatis
    dan mengurutkannya secara alfabetis (case-sensitive, huruf kecil/besar rapi).
    """
    return sorted(ast_root.get_variables(), key=lambda v: (v.lower(), v))


def collect_unique_subexpressions(ast_root: ASTNode) -> List[ASTNode]:
    """
    Mengumpulkan sub-ekspresi bertahap (termasuk ekspresi akhir) tanpa duplikasi
    untuk ditampilkan sebagai kolom-kolom tabel kebenaran.
    """
    raw_subs = ast_root.get_subexpressions()
    unique_subs: List[ASTNode] = []
    seen_labels: Set[str] = set()
    for sub in raw_subs:
        label = sub.to_canonical_str()
        if label not in seen_labels:
            seen_labels.add(label)
            unique_subs.append(sub)
    return unique_subs


def generate_truth_combinations(
    variables: List[str],
    descending: bool = True,
) -> List[Dict[str, bool]]:
    """
    Membangkitkan seluruh 2^n kombinasi nilai kebenaran untuk daftar variabel.
    Jika `descending=True`, urutan dimulai dari True (T) ke False (F) sesuai
    konvensi buku teks Logika Matematika / Matematika Diskrit.
    """
    n = len(variables)
    domain = [True, False] if descending else [False, True]
    combinations: List[Dict[str, bool]] = []
    for values in itertools.product(domain, repeat=n):
        combinations.append(dict(zip(variables, values)))
    return combinations


# ==============================================================================
# 5. EVALUASI & FORMATTING TABEL KEBENARAN RAPI
# ==============================================================================

def format_bool(val: bool, style: str = "TF") -> str:
    """Mengubah nilai boolean Python menjadi string tampilan ('T'/'F' atau '1'/'0')."""
    if style == "10":
        return "1" if val else "0"
    return "T" if val else "F"


def render_box_table(headers: List[str], rows: List[List[str]]) -> str:
    """
    Merender tabel kebenaran dengan garis border Unicode yang presisi dan rata tengah.
    Tidak bergantung pada library eksternal sehingga tampilan selalu konsisten.
    """
    col_widths = [
        max(len(str(headers[col_idx])), *(len(str(row[col_idx])) for row in rows))
        if rows else len(str(headers[col_idx]))
        for col_idx in range(len(headers))
    ]

    def make_border(left: str, mid: str, right: str, fill: str = "─") -> str:
        segments = [fill * (w + 2) for w in col_widths]
        return f"{left}{mid.join(segments)}{right}"

    top_border = make_border("┌", "┬", "┐", "─")
    header_sep = make_border("╞", "╪", "╡", "═")
    row_sep    = make_border("├", "┼", "┤", "─")
    bot_border = make_border("└", "┴", "┘", "─")

    lines = [top_border]
    header_cells = [f" {headers[i].center(col_widths[i])} " for i in range(len(headers))]
    lines.append(f"│{'│'.join(header_cells)}│")
    lines.append(header_sep)

    for r_idx, row in enumerate(rows):
        cells = [f" {str(row[i]).center(col_widths[i])} " for i in range(len(headers))]
        lines.append(f"│{'│'.join(cells)}│")
        if r_idx < len(rows) - 1:
            lines.append(row_sep)

    lines.append(bot_border)
    return "\n".join(lines)


def classify_proposition(final_results: List[bool]) -> Tuple[str, str]:
    """
    Mengklasifikasikan hasil akhir ekspresi logika ke dalam:
    - Tautologi (selalu bernilai True / T)
    - Kontradiksi (selalu bernilai False / F)
    - Kontingensi (memiliki nilai True dan False)
    """
    if all(final_results):
        return "TAUTOLOGI", "Selalu bernilai BENAR (T) untuk seluruh kombinasi."
    if not any(final_results):
        return "KONTRADIKSI", "Selalu bernilai SALAH (F) untuk seluruh kombinasi."
    true_count = sum(1 for r in final_results if r)
    false_count = len(final_results) - true_count
    return (
        "KONTINGENSI",
        f"Bernilai BENAR (T) pada {true_count} baris dan SALAH (F) pada {false_count} baris.",
    )


def analyze_and_print_truth_table(
    expression: str,
    show_subexpressions: bool = True,
    bool_style: str = "TF",
) -> Dict[str, object]:
    """
    Pipeline lengkap PJBL-1 Sesi ke-2:
    Input -> Tokenisasi -> Parsing -> Deteksi Variabel -> Kombinasi 2^n -> Evaluasi & Tabel.
    """
    print("=" * 78)
    print(f"EKSPRESI INPUT : {expression}")
    print("=" * 78)

    # Tahap 1 & 2: Tokenisasi dan Parsing
    tokens, ast_root, parser_engine = parse_expression(expression)
    token_str = ", ".join(f"{t.type}('{t.value}')" for t in tokens)

    # Tahap 3: Deteksi Variabel Otomatis
    variables = detect_variables(ast_root)
    num_vars = len(variables)
    num_rows = 2 ** num_vars

    # Kumpulkan sub-ekspresi untuk kolom tabel
    subexpressions = collect_unique_subexpressions(ast_root) if show_subexpressions else [ast_root]
    if not subexpressions:
        subexpressions = [ast_root]

    print(f"[1] Mesin Parser       : {parser_engine}")
    print(f"[2] Hasil Tokenisasi   : {len(tokens)} token -> [{token_str}]")
    print(f"[3] Bentuk Kanonik AST : {ast_root.to_canonical_str()}")
    print(f"[4] Variabel Terdeteksi: {variables if variables else '(Konstanta saja)'} (n = {num_vars})")
    print(f"[5] Jumlah Kombinasi   : 2^{num_vars} = {num_rows} baris kombinasi")
    print("-" * 78)

    # Tahap 4: Bangkitkan 2^n kombinasi nilai kebenaran
    combinations = generate_truth_combinations(variables, descending=True)

    # Tahap 5: Evaluasi tiap kombinasi & bangun baris tabel
    headers = ["No"] + variables + [sub.to_canonical_str() for sub in subexpressions]
    table_rows: List[List[str]] = []
    final_results: List[bool] = []

    for idx, env in enumerate(combinations, start=1):
        row_cells = [str(idx)]
        for var in variables:
            row_cells.append(format_bool(env[var], style=bool_style))
        for sub in subexpressions:
            val = sub.evaluate(env)
            row_cells.append(format_bool(val, style=bool_style))
        final_val = ast_root.evaluate(env)
        final_results.append(final_val)
        table_rows.append(row_cells)

    # Tampilkan Tabel Kebenaran
    print(render_box_table(headers, table_rows))

    # Ringkasan & Klasifikasi Proposisi
    kategori, penjelasan = classify_proposition(final_results)
    print(f"Kesimpulan Sifat Ekspresi: {kategori} — {penjelasan}")
    print("=" * 78 + "\n")

    return {
        "expression": expression,
        "canonical": ast_root.to_canonical_str(),
        "tokens": tokens,
        "variables": variables,
        "combinations_count": num_rows,
        "final_results": final_results,
        "classification": kategori,
    }


# ==============================================================================
# 6. PENGUJIAN CONTOH EKSPRESI & MODE INTERAKTIF CLI
# ==============================================================================

def run_demonstration_tests() -> None:
    """
    Menjalankan pengujian otomatis dengan 3 ekspresi contoh yang berbeda
    (memenuhi syarat Milestone Pertemuan 3 poin ke-6: minimal 2 ekspresi berbeda).
    """
    sample_expressions = [
        # Contoh 1: Sesuai soal modul PJBL-1 Sesi 2 (3 variabel, Kontingensi)
        "p AND (q OR NOT r)",
        # Contoh 2: Hukum De Morgan / Ekuivalensi Logika (2 variabel, Tautologi)
        "NOT (p AND q) <-> (NOT p OR NOT q)",
        # Contoh 3: Implikasi & Biimplikasi dengan 3 variabel (Silogisme Hipotetis, Tautologi)
        "((p -> q) AND (q -> r)) -> (p -> r)",
    ]

    print("\n" + "#" * 78)
    print("   DEMO OTOMATIS PJBL-1 SESI KE-2: PARSER & GENERATOR TABEL KEBENARAN")
    print("#" * 78 + "\n")

    for expr in sample_expressions:
        analyze_and_print_truth_table(expr, show_subexpressions=True)


def interactive_cli() -> None:
    """Mode interaktif agar pengguna/dosen dapat menguji ekspresi bebas secara langsung."""
    print("Panduan Operator yang Didukung:")
    print("  - Negasi (NOT)     : NOT, ~, !, ¬")
    print("  - Konjungsi (AND)  : AND, &, ^, ∧")
    print("  - Disjungsi (OR)   : OR, |, ∨")
    print("  - Eksklusif OR     : XOR, ⊕")
    print("  - Implikasi        : IMPLIES, IMP, ->, =>, →")
    print("  - Biimplikasi      : IFF, <->, <=>, ↔")
    print("Ketik 'keluar', 'exit', atau tekan Enter kosong untuk mengakhiri program.\n")

    while True:
        try:
            user_input = input("Masukkan ekspresi logika > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nProgram selesai. Terima kasih!")
            break

        if not user_input or user_input.lower() in ("q", "quit", "exit", "keluar"):
            print("Program selesai. Terima kasih!")
            break

        try:
            analyze_and_print_truth_table(user_input, show_subexpressions=True)
        except (SyntaxError, ValueError) as err:
            print(f"[ERROR INPUT] {err}\nSilakan periksa kembali penulisan ekspresi Anda.\n")


if __name__ == "__main__":
    # Jika argumen CLI diberikan (misal: python3 truth_table_generator.py "p AND q")
    if len(sys.argv) > 1:
        custom_expr = " ".join(sys.argv[1:])
        analyze_and_print_truth_table(custom_expr, show_subexpressions=True)
    else:
        run_demonstration_tests()
        if sys.stdin.isatty():
            interactive_cli()