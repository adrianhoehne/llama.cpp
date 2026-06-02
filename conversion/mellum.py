from __future__ import annotations

import json

from .base import ModelBase, gguf, logger
from .qwen import Qwen3MoeModel


@ModelBase.register("MellumForCausalLM")
class MellumModel(Qwen3MoeModel):
    model_arch = gguf.MODEL_ARCH.MELLUM

    def set_vocab(self):
        self._set_vocab_mellum()

    def _set_vocab_mellum(self):
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(self.dir_model / "tokenizer.json"))
        vocab = tokenizer.get_vocab()
        vocab_size = self.hparams.get("vocab_size", len(vocab))
        assert max(vocab.values()) < vocab_size

        tokpre = self._get_vocab_pre()
        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in vocab.items()}
        added_vocab = tokenizer.get_added_vocab()
        added_tokens_decoder = tokenizer.added_tokens_decoder

        tokens: list[str] = []
        toktypes: list[int] = []
        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
                continue

            token: str = reverse_vocab[i]
            if token in added_vocab:
                if not added_tokens_decoder[i].normalized:
                    previous_token = token
                    token = tokenizer.decode(tokenizer.encode(token, add_special_tokens=False))
                    if previous_token != token:
                        logger.info(f"{repr(previous_token)} is encoded and decoded back to {repr(token)} using PreTrainedTokenizerFast")

                if added_tokens_decoder[i].special or self.does_token_look_special(token):
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    token = token.replace(b"\xe2\x96\x81".decode("utf-8"), " ")
                    toktypes.append(gguf.TokenType.USER_DEFINED)
            else:
                toktypes.append(gguf.TokenType.NORMAL)

            tokens.append(token)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _get_vocab_pre(self):
        with open(self.dir_model / "tokenizer.json", "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)

        pre_tokenizer = tokenizer_config.get("pre_tokenizer")
        if pre_tokenizer == {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Digits", "individual_digits": True},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
            ],
        }:
            return "starcoder"

        return "mellum"

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        if (sliding_window := self.hparams.get("sliding_window")) is not None:
            self.gguf_writer.add_sliding_window(sliding_window)

            if layer_types := self.hparams.get("layer_types"):
                self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in layer_types])
            else:
                self.gguf_writer.add_sliding_window_pattern(4)

        if (norm_topk_prob := self.hparams.get("norm_topk_prob")) is not None:
            self.gguf_writer.add_expert_weights_norm(norm_topk_prob)

        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
