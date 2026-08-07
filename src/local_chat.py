#!/usr/bin/env python3
"""Interactive local sampler for the final OpenHermes SFT checkpoint.

Designed for a memory-constrained Apple-silicon Mac: one model process, MPS
when available, no compilation, and the explicit SDPA reference attention path.
It is a convenience sampler, not a performance or benchmark implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from sample import load_model_and_tokenizer, resolve_device, sample_next_token, set_seed


DEFAULT_SYSTEM = "You are a helpful, honest assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat locally with the final SFT checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-context-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def fit_context(tokenizer: Any, messages: list[dict[str, str]], limit: int) -> str:
    """Drop the oldest complete user/assistant turn before exceeding local limit."""
    while True:
        prompt = render_prompt(tokenizer, messages)
        if len(tokenizer.encode(prompt, add_special_tokens=False)) <= limit:
            return prompt
        # Keep the system message, then discard one oldest conversational message.
        if len(messages) <= 2:
            raise ValueError("one message exceeds --max-context-tokens; shorten it")
        messages.pop(1)


@torch.no_grad()
def generate_reply(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    """Return generated tokens only; never slice decoded text by character count."""
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    if input_ids.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus reply exceeds the model context length")
    output, cache = model.forward_cached(input_ids, return_logits=True)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_token = sample_next_token(
            output["logits"][:, -1, :].float(),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        token = int(next_token.item())
        if tokenizer.eos_token_id is not None and token == int(tokenizer.eos_token_id):
            break
        generated.append(token)
        output, cache = model.forward_cached(next_token.to(device=device), cache=cache, return_logits=True)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    if args.max_context_tokens < 128:
        raise ValueError("--max-context-tokens must be at least 128")
    device = resolve_device(args.device)
    set_seed(args.seed)
    checkpoint = args.checkpoint.expanduser().resolve()
    model, tokenizer, _, config = load_model_and_tokenizer(
        checkpoint,
        run_dir=None,
        tokenizer_name=None,
        tokenizer_revision=None,
        # FA3/Flex are CUDA paths.  This intentionally selects the portable
        # oracle implementation rather than silently trying a GPU-only kernel.
        attention_backend="sdpa_reference",
        device=device,
        dtype="fp32",
    )
    context_limit = min(args.max_context_tokens, int(config.max_seq_len) - args.max_new_tokens)
    if context_limit <= 0:
        raise ValueError("max-new-tokens leaves no context room")

    print(f"Loaded {checkpoint.name} on {device.type}. Context limit: {context_limit} tokens.")
    print("Commands: /reset, /system <text>, /quit")
    messages: list[dict[str, str]] = [{"role": "system", "content": DEFAULT_SYSTEM}]
    while True:
        try:
            user_text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not user_text:
            continue
        if user_text == "/quit":
            print("bye")
            return
        if user_text == "/reset":
            messages = [{"role": "system", "content": DEFAULT_SYSTEM}]
            print("conversation reset")
            continue
        if user_text.startswith("/system "):
            messages = [{"role": "system", "content": user_text.removeprefix("/system ").strip()}]
            print("system prompt replaced; conversation reset")
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            prompt = fit_context(tokenizer, messages, context_limit)
            answer = generate_reply(
                model,
                tokenizer,
                prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        except Exception as exc:
            messages.pop()
            print(f"generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        messages.append({"role": "assistant", "content": answer})
        print(f"\nassistant> {answer}")
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
