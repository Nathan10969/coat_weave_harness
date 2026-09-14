"""Qwen3 reranker used by the patent retrieval pipeline."""

from __future__ import annotations

from typing import Sequence


class QwenReranker:
    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        max_length: int = 4096,
        dtype=None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left", trust_remote_code=True
        )
        model_kwargs = {"trust_remote_code": True}
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            .eval()
            .to(device)
        )

        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the "
            "requirements based on the Query and the Instruct provided. "
            "The answer can only be yes or no.<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def __call__(
        self,
        queries: Sequence[str],
        documents: Sequence[str],
        instruction: str = "Retrieve patent documents that answer the technical query",
    ) -> list[float]:
        if len(queries) != len(documents):
            raise ValueError("queries and documents must have the same length")
        pairs = [
            f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
            for query, document in zip(queries, documents)
        ]
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens),
        )
        inputs["input_ids"] = [
            self.prefix_tokens + token_ids + self.suffix_tokens
            for token_ids in inputs["input_ids"]
        ]
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        inputs = {name: value.to(self.model.device) for name, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[:, -1, :]
            binary_logits = self.torch.stack(
                [logits[:, self.false_id], logits[:, self.true_id]], dim=1
            )
            return self.torch.softmax(binary_logits, dim=1)[:, 1].float().cpu().tolist()
