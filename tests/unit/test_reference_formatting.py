from app.reference_formatting import preserves_non_whitespace


def test_reformatting_keeps_symbols_and_numbers_in_order():
    source = "operator **  alias ^  precedence 2"
    assert preserves_non_whitespace(source, "operator **\nalias ^\nprecedence 2")
    assert not preserves_non_whitespace(source, "operator *\nalias ^\nprecedence 2")
    assert not preserves_non_whitespace(source, "operator **\nalias ^\nprecedence 3")
    assert not preserves_non_whitespace(source, "alias ^\noperator **\nprecedence 2")


def test_empty_content_is_not_accepted():
    assert not preserves_non_whitespace("", "")
