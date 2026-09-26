"""Tests for the deterministic filler and stutter strip."""
import pytest
from src.text_postprocess import has_filler_words, strip_filler_words


# --- Repeated words that are content, not stutters ------------------------

def test_dictated_phone_number_keeps_every_digit():
    # Regression: the stutter rule collapsed "zero zero" and "zero zero
    # zero", so the pasted number was three digits short.
    text = "my number is one three eight zero zero one three eight zero zero zero"
    assert strip_filler_words(text) == text
    assert not has_filler_words(text)


@pytest.mark.parametrize("text", [
    "call me at five five five one two one two",
    "my pin is four four four four",
    "the gate code is nine nine one one",
    "room two oh oh five",
    "agent double oh seven seven",
    "the year twenty twenty",
    "we land at ten ten",
    "the code is 1 1 2 3",
])
def test_repeated_english_numbers_are_kept(text):
    assert strip_filler_words(text) == text
    assert not has_filler_words(text)


@pytest.mark.parametrize("word", [
    "zero", "oh", "o", "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
    "twenty", "hundred", "thousand", "double", "triple",
])
def test_no_number_word_is_collapsed(word):
    text = f"the code is {word} {word} {word}"
    assert strip_filler_words(text) == text


def test_fillers_inside_a_number_go_but_the_digits_stay():
    text = "um my number is one one two uh two"
    assert strip_filler_words(text) == "my number is one one two two"


@pytest.mark.parametrize("text", [
    "一三八零零一三八零零零",
    "我的电话是一三八零零一三八零零零",
    "我的电话是幺三八零零幺三八零零零",
    "验证码是八八八八",
    "我的电话是138 0013 8000",
])
def test_chinese_numbers_are_kept(text):
    assert strip_filler_words(text) == text
    assert not has_filler_words(text)


def test_chinese_hesitation_goes_but_the_digits_stay():
    text = "嗯，我的电话是一三八零零一三八零零零"
    assert strip_filler_words(text) == "我的电话是一三八零零一三八零零零"


@pytest.mark.parametrize("text", [
    # Emphasis is what the speaker said, not a disfluency. The LLM step
    # may still tidy it with context; the rule-strip must not.
    "it's very very good",
    "no no no that's wrong",
    "a long long time ago",
    # Reduplications and names
    "bye bye",
    "the food was so so",
    "we flew to Walla Walla",
    "Fei Fei will join us",
    # Grammatical doubles, and one clause ending where the next begins
    "she had had enough",
    "I know that that is true",
    "I told you you were right",
    "I love it it's great",
    # Spelled letters and grades
    "my name is J O H N N Y",
    "I got A A B in my exams",
])
def test_deliberate_repeats_are_kept(text):
    assert strip_filler_words(text) == text
    assert not has_filler_words(text)


# --- Real stutters are still collapsed ------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("I I I think so", "I think so"),
    ("the the cat", "the cat"),
    ("The the cat sat down", "The cat sat down"),
    ("we we should go", "we should go"),
    ("there's a a problem", "there's a problem"),
    ("A a problem came up", "A problem came up"),
    ("we we're late", "we're late"),
    ("and and then we left", "and then we left"),
    ("I want to to leave", "I want to leave"),
    ("I um I think", "I think"),
    ("so the the number is five five five", "so the number is five five five"),
])
def test_stutters_are_collapsed(text, expected):
    assert strip_filler_words(text) == expected
    assert has_filler_words(text)
