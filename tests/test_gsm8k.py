from greenroute import gsm8k


def test_gold_answer_strips_formatting():
    assert gsm8k.gold_answer("Some work\n#### 1,234") == "1234"


def test_extract_prefers_answer_line():
    text = "First 3 apples, then 5 more.\nAnswer: 8"
    assert gsm8k.extract_answer(text) == "8"


def test_extract_handles_bold_dollar_and_decimals():
    assert gsm8k.extract_answer("Answer: **$18.00**") == "18"
    assert gsm8k.extract_answer("Answer: 2.5") == "2.5"


def test_extract_falls_back_to_last_number():
    assert gsm8k.extract_answer("so the total is 42.") == "42"
    assert gsm8k.extract_answer("no digits here") is None


def test_is_correct():
    assert gsm8k.is_correct("...\nAnswer: 72", "work\n#### 72")
    assert not gsm8k.is_correct("...\nAnswer: 71", "work\n#### 72")
