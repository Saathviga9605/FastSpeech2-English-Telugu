"""Per-language symbol tables. Unlike upstream, an unknown symbol is an error."""
SILENCES = ["sp", "spn", "sil"]
PAD = "_"


class UnknownSymbolError(ValueError):
    pass


class Vocabulary:
    def __init__(self, symbols):
        if symbols[0] != PAD or len(set(symbols)) != len(symbols):
            raise ValueError("Symbols must start with the pad symbol and be unique")
        self.symbols = list(symbols)
        self.to_id = {s: i for i, s in enumerate(symbols)}

    def __len__(self):
        return len(self.symbols)

    def encode(self, phones):
        unknown = sorted({p for p in phones if p not in self.to_id})
        if unknown:
            raise UnknownSymbolError("Symbols not in vocabulary: {}".format(unknown))
        return [self.to_id[p] for p in phones]

    def decode(self, ids):
        return [self.symbols[i] for i in ids]
