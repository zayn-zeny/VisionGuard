"""Tiny test file for the Task 6.6 red-then-green CI exercise."""


def add(a: int, b: int) -> int:
    return a + b


def slugify(text: str) -> str:
    return "-".join(text.lower().split())


def test_add():
    assert add(2, 3) == 5


def test_slugify():
    assert slugify("AI vs Real") == "ai-vs-real"
