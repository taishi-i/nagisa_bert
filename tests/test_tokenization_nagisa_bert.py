from nagisa_bert.tokenization_nagisa_bert import NagisaBertTokenizer


class TestNagisaBertTokenizer:

    tokenizer = NagisaBertTokenizer.from_pretrained("taishi-i/nagisa_bert")

    text = "nagisaで[MASK]できるBERTモデルです"

    def test_vocab_size(self):
        expected_vocab_size = 32000
        assert expected_vocab_size == self.tokenizer.vocab_size

    def test_get_vocab(self):
        expected_vocab_size = 32000
        expected_internet_token_id = 10171

        assert expected_vocab_size == len(self.tokenizer.get_vocab())
        assert expected_internet_token_id == self.tokenizer.get_vocab()["インターネット"]

    def test_pad_token(self):
        expected_pad_token = "[PAD]"
        expected_pad_token_id = 0

        assert expected_pad_token == self.tokenizer.pad_token
        assert expected_pad_token_id == self.tokenizer.pad_token_id

    def test_unk_token(self):
        expected_unk_token = "[UNK]"
        expected_unk_token_id = 1

        assert expected_unk_token == self.tokenizer.unk_token
        assert expected_unk_token_id == self.tokenizer.unk_token_id

    def test_cls_token(self):
        expected_cls_token = "[CLS]"
        expected_cls_token_id = 2

        assert expected_cls_token == self.tokenizer.cls_token
        assert expected_cls_token_id == self.tokenizer.cls_token_id

    def test_sep_token(self):
        expected_sep_token = "[SEP]"
        expected_sep_token_id = 3

        assert expected_sep_token == self.tokenizer.sep_token
        assert expected_sep_token_id == self.tokenizer.sep_token_id

    def test_mask_token(self):
        expected_mask_token = "[MASK]"
        expected_mask_token_id = 4

        assert expected_mask_token == self.tokenizer.mask_token
        assert expected_mask_token_id == self.tokenizer.mask_token_id

    def test_tokenize_basic(self):
        expected_tokens = [
            "na",
            "##g",
            "##is",
            "##a",
            "で",
            "[MASK]",
            "できる",
            "BE",
            "##R",
            "##T",
            "モデル",
            "です",
        ]

        tokens = self.tokenizer.tokenize(self.text)
        assert expected_tokens == tokens

    def test_tokenize_sample_hankaku_zenkaku(self):
        text = "ｺﾝﾊﾞﾝﾊ１２３４５"
        expected_tokens = ["コン", "##バン", "##ハ", "1", "2", "3", "4", "5"]

        tokens = self.tokenizer.tokenize(text)
        assert expected_tokens == tokens

    def test_tokenize_sample_emoji_zenkaku(self):
        text = "こんばんは😀ＰＹＴＨＯＮ"
        expected_tokens = ["こん", "##ばん", "##は", "[UNK]", "P", "##Y", "##TH", "##ON"]

        tokens = self.tokenizer.tokenize(text)
        assert expected_tokens == tokens

    def test_tokenize_sample_kaomoji(self):
        text = "(人•ᴗ•♡)こんばんは♪"
        expected_tokens = ["[UNK]", "こん", "##ばん", "##は", "♪"]

        tokens = self.tokenizer.tokenize(text)
        assert expected_tokens == tokens

    def test_tokenize_unknown_chars(self):
        text = "𪗱𪘂𪘚𪚲"
        expected_tokens = ["[UNK]", "[UNK]", "[UNK]", "[UNK]"]

        tokens = self.tokenizer.tokenize(text)
        assert expected_tokens == tokens

    def test_encode(self):
        expected_output = {
            "input_ids": [
                2,
                30314,
                4015,
                8143,
                4012,
                451,
                4,
                8097,
                14270,
                4119,
                4121,
                8455,
                8489,
                3,
            ],
            "token_type_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }

        output = self.tokenizer(self.text)
        assert expected_output["input_ids"] == output["input_ids"]
        assert expected_output["token_type_ids"] == output["token_type_ids"]
        assert expected_output["attention_mask"] == output["attention_mask"]

    def test_convert_ids_to_tokens(self):
        output = self.tokenizer(self.text)

        expected_tokens = [
            "[CLS]",
            "na",
            "##g",
            "##is",
            "##a",
            "で",
            "[MASK]",
            "できる",
            "BE",
            "##R",
            "##T",
            "モデル",
            "です",
            "[SEP]",
        ]

        tokens = self.tokenizer.convert_ids_to_tokens(output["input_ids"])
        assert expected_tokens == tokens
