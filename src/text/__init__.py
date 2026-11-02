"""Language frontends."""
from .vocab import UnknownSymbolError, Vocabulary


def get_frontend(language, preprocess_config=None):
    """Return a callable (text, strict=False) -> FrontendResult."""
    if language == "telugu":
        from . import telugu

        return telugu.process
    if language == "english":
        from .english import EnglishFrontend

        return EnglishFrontend(preprocess_config["path"]["lexicon_path"]).process
    raise ValueError("Unknown language {!r}".format(language))
