"""Tests for channel stoplists."""

from nlp_graph.extract.channel_stoplist import channel_stoplist, has_content_token, is_junk_phrase, strip_channel_phrases


def test_linkedin_boilerplate_stripped():
    text = "Excited to share our thought leadership on lakehouse architecture"
    out = strip_channel_phrases(text, "linkedin")
    assert "excited to share" not in out.lower()


def test_junk_phrase_rejects_generic():
    stops = channel_stoplist("reddit")
    assert is_junk_phrase("really good", stops) is True


def test_junk_phrase_keeps_domain():
    stops = channel_stoplist("reddit")
    assert is_junk_phrase("consumption pricing", stops) is False


def test_junk_phrase_rejects_light_verbs():
    stops = channel_stoplist("reddit")
    for phrase in ("designed to", "able to", "kind of", "game-changer for", "about the", "part of"):
        assert is_junk_phrase(phrase, stops) is True


def test_has_content_token_rejects_conversational():
    stops = channel_stoplist("reddit")
    for phrase in ("worth it", "scale it", "better than", "handle the", "worried about"):
        assert has_content_token(phrase, phrase.lower().split(), stops) is False


def test_has_content_token_keeps_domain():
    stops = channel_stoplist("reddit")
    assert has_content_token("migration from snowflake", ["migration", "from", "snowflake"], stops) is True
    assert has_content_token("dbt cloud pricing", ["dbt", "cloud", "pricing"], stops) is True


def test_junk_phrase_keeps_migration():
    stops = channel_stoplist("reddit")
    assert is_junk_phrase("migration from snowflake", stops) is False
