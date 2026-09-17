import unittest

from tokenizer import CharacterTokenizer


class TokenizerTests(unittest.TestCase):
    def test_repeatable_mapping(self):
        self.assertEqual(CharacterTokenizer.from_text("cat").characters, ("a", "c", "t"))
        self.assertEqual(CharacterTokenizer.from_text("tac").characters, ("a", "c", "t"))

    def test_round_trip_including_unicode_spaces_and_newlines(self):
        text = "café cat\ncat 🐈"
        tokenizer = CharacterTokenizer.from_text(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_shifted_targets(self):
        tokenizer = CharacterTokenizer.from_text("cat")
        self.assertEqual(tokenizer.training_pair("cat"), ([1, 0], [0, 2]))

    def test_unknown_character_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            CharacterTokenizer.from_text("cat").encode("dog")

    def test_empty_vocabulary_rejected(self):
        with self.assertRaises(ValueError):
            CharacterTokenizer.from_text("")

    def test_short_training_examples_rejected(self):
        tokenizer = CharacterTokenizer.from_text("cat")
        for text in ("", "c"):
            with self.assertRaises(ValueError):
                tokenizer.training_pair(text)

    def test_invalid_token_ids_rejected(self):
        tokenizer = CharacterTokenizer.from_text("cat")
        for ids in ([-1], [3], [True], [1.0], ["1"]):
            with self.assertRaises(ValueError):
                tokenizer.decode(ids)

    def test_duplicate_vocabulary_rejected(self):
        with self.assertRaises(ValueError):
            CharacterTokenizer(("a", "a"))


if __name__ == "__main__":
    unittest.main()
