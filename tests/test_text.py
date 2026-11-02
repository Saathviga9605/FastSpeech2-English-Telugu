import pytest

from src.text import telugu
from src.text.vocab import UnknownSymbolError, Vocabulary


@pytest.mark.parametrize(
    "word,phones",
    [
        ("అమ్మ", "a m m a"),
        ("రాముడు", "r aa m u dd u"),
        ("తెలుగు", "t e l u g u"),
        ("సంగీతం", "s a ng g ii t a m"),
        ("పంపు", "p a m p u"),
        ("దుఃఖం", "d u h kh a m"),
        ("బ్రహ్మ", "b r a h m a"),
        ("క్యూ", "k y uu"),
    ],
)
def test_word_rules(word, phones):
    assert telugu.word_to_phones(word) == phones.split()


def test_every_phone_is_in_vocabulary():
    vocab = Vocabulary(telugu.phone_symbols())
    vocab.encode(telugu.process("నరకవాసి. మిత్రవిందకుడు, 25 AB").phones)


def test_two_part_vowel_sign_is_canonicalized():
    assert telugu.process("కై").phones == telugu.process("కై").phones


def test_strict_rejects_unknown_character():
    with pytest.raises(ValueError):
        telugu.process("తెలుగు ☃", strict=True)


def test_non_strict_reports_unknown_character_and_keeps_original():
    result = telugu.process("తెలుగు ☃ భాష")
    assert result.original == "తెలుగు ☃ భాష"
    assert "sp" in result.phones
    assert any("U+2603" in issue for issue in result.issues)


def test_misplaced_sign_is_reported_not_dropped():
    result = telugu.process("ాక")
    assert any("misplaced" in issue for issue in result.issues)


def test_empty_and_symbol_only_input_do_not_crash():
    assert telugu.process("").phones == []
    result = telugu.process("@#$")
    assert result.phones == []
    assert len(result.issues) == 3


def test_vocabulary_raises_instead_of_dropping():
    with pytest.raises(UnknownSymbolError):
        Vocabulary(["_", "a"]).encode(["a", "b"])


def test_assignment_sentences_are_preserved_and_complete():
    import json

    from src.inference.synthesize import DEFAULT_SENTENCES

    data = json.loads(DEFAULT_SENTENCES.read_text(encoding="utf-8"))
    assert [s["id"] for s in data["english"]] == ["E1", "E2a", "E2b", "E3", "E4"]
    assert [s["id"] for s in data["telugu"]] == ["T1", "T2a", "T2b", "T3", "T4"]
    assert "rhyth  m" in data["english"][3]["text"]
    assert "అడ  విలో" in data["telugu"][3]["text"]


@pytest.mark.parametrize(
    "value,words",
    [(12, "పన్నెండు"), (45, "నలభై ఐదు"), (100, "వంద"), (2026, "రెండు వేల ఇరవై ఆరు"), (1000, "వెయ్యి")],
)
def test_telugu_numbers(value, words):
    assert telugu.number_to_words(value) == words


def test_telugu_t4_numbers_and_original_kept():
    text = "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు"
    result = telugu.process(text)
    assert result.original == text
    assert "పన్నెండవ" in result.normalized and "పది నలభై ఐదు" in result.normalized
    assert "రెండు వేల ఇరవై ఆరు" in result.normalized
    assert sum("read" in issue for issue in result.issues) == 3
    assert "0" not in result.normalized


def test_detached_vowel_sign_is_read_and_logged():
    result = telugu.process("వచ్చ ింది")
    assert result.phones == "v a c c a i n d i".split()
    assert any("misplaced" in issue for issue in result.issues)


def test_english_time_and_meridiem():
    from src.config import load_configs
    from src.text.english import EnglishFrontend

    frontend = EnglishFrontend(load_configs("english")[0]["path"]["lexicon_path"])
    result = frontend.process(
        "The meeting will start at 10:45 AM on 12 November 2026", resolve_homographs=False
    )
    assert "ten forty-five a m on twelve november twenty twenty-six" in result.normalized.replace("  ", " ")


def _load_sentences():
    import json

    from src.inference.synthesize import DEFAULT_SENTENCES

    return json.loads(DEFAULT_SENTENCES.read_text(encoding="utf-8"))


def test_every_sentence_records_all_versions():
    data = _load_sentences()
    for sentence in data["english"] + data["telugu"]:
        assert set(sentence) == {"id", "text", "pdf_rendered", "pdf_text_layer", "message_text"}
        assert sentence["text"] == sentence["pdf_rendered"]


def test_selected_text_matches_pdf_layer_apart_from_spaces():
    data = _load_sentences()
    for sentence in data["english"] + data["telugu"]:
        assert sentence["text"].replace(" ", "") == sentence["pdf_text_layer"].replace(" ", "")


def test_run_of_spaces_is_logged_and_original_kept():
    data = _load_sentences()
    t3 = data["telugu"][3]["text"]
    result = telugu.process(t3)
    assert result.original == t3
    assert any("run of spaces" in issue for issue in result.issues)
