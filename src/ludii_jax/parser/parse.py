"""
Ludii .lud file parser.

Uses the permissive Lark grammar that accepts 100% of Ludii games (1,212/1,212).
Returns a Lark parse tree that downstream analysis and compilation modules walk.
"""

import os
from lark import Lark, Tree, Token

_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "ludii_grammar.lark")
_parser = None


def get_parser():
    global _parser
    if _parser is None:
        _parser = Lark.open(_GRAMMAR_PATH, start="game", parser="earley", keep_all_tokens=True)
    return _parser


def parse(lud_text: str) -> Tree:
    """Parse Ludii .lud text into a parse tree."""
    return get_parser().parse(lud_text)


def find_child(tree: Tree, name: str):
    """Find the first child tree with the given rule name."""
    for c in tree.children:
        if isinstance(c, Tree) and c.data == name:
            return c
    return None


def find_all(tree: Tree, name: str) -> list:
    """Find all children with the given rule name."""
    return [c for c in tree.children if isinstance(c, Tree) and c.data == name]


def get_text(node) -> str:
    """Get all text content from a tree/token, recursively. Skips parens/braces."""
    if isinstance(node, Token):
        s = str(node)
        if s in ("(", ")", "{", "}"):
            return ""
        return s
    if isinstance(node, Tree):
        return " ".join(p for p in (get_text(c) for c in node.children) if p).strip()
    return str(node)


def get_string_token(tree: Tree) -> str:
    """Extract the first ESCAPED_STRING token value from a tree."""
    for c in tree.children:
        if isinstance(c, Token) and c.type == "ESCAPED_STRING":
            return str(c).strip('"')
    return ""
