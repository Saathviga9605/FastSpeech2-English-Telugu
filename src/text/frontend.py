"""Shared result type for the language frontends."""
from dataclasses import dataclass, field


@dataclass
class FrontendResult:
    original: str  # the input exactly as given
    normalized: str = ""
    phones: list = field(default_factory=list)
    words: list = field(default_factory=list)  # (token, [phones]) pairs for alignment
    issues: list = field(default_factory=list)  # every replacement or removal made

    def as_dict(self):
        return {
            "original": self.original,
            "normalized": self.normalized,
            "phones": self.phones,
            "issues": self.issues,
        }
