"""Lesson 1: a character tokenizer. No model or external dependencies yet.

Token IDs are labels, not meaning or rankings. Real LLMs usually use more
complex tokenizers; one character per token makes this first lesson visible.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterTokenizer:
    characters: tuple[str, ...]

    def __post_init__(self):
        if not self.characters or any(len(c) != 1 for c in self.characters):
            raise ValueError("Vocabulary must contain individual characters.")
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("Vocabulary cannot contain duplicate characters.")

    @classmethod
    def from_text(cls, text: str):
        # A fixed sorted vocabulary makes the mapping repeatable.
        return cls(tuple(sorted(set(text))))

    def encode(self, text: str) -> list[int]:
        lookup = {character: token_id for token_id, character in enumerate(self.characters)}
        unknown = sorted(set(text) - set(lookup))
        if unknown:
            raise ValueError(f"Characters outside this vocabulary: {unknown!r}")
        return [lookup[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        if any(type(i) is not int or not 0 <= i < len(self.characters) for i in token_ids):
            raise ValueError("Each token ID must be an integer in the vocabulary range.")
        return "".join(self.characters[i] for i in token_ids)

    def training_pair(self, text: str) -> tuple[list[int], list[int]]:
        if len(text) < 2:
            raise ValueError("Use at least two characters to make next-character targets.")
        token_ids = self.encode(text)
        # At each position, the answer is the character one position ahead.
        return token_ids[:-1], token_ids[1:]


if __name__ == "__main__":
    tokenizer = CharacterTokenizer.from_text("cat")
    inputs, targets = tokenizer.training_pair("cat")
    print("LESSON 1 — Text becomes numbers")
    print("Vocabulary:", {c: i for i, c in enumerate(tokenizer.characters)})
    print("Text: cat")
    print("Token IDs:", tokenizer.encode("cat"))
    print("Decoded again:", tokenizer.decode(tokenizer.encode("cat")))
    print("Input positions:", tokenizer.decode(inputs), inputs)
    print("Target positions:", tokenizer.decode(targets), targets)
    print("First prediction: given c, predict a.")
    print("Second prediction: given ca, predict t (a causal model can use earlier context).")
    print("These are training examples, not predictions from a trained model yet.")
