from __future__ import annotations

import math

import torch


def sample_next_token(logits, *, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError("sampling requires finite one-dimensional logits")
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("invalid sampling parameters")
    if temperature == 0:
        return int(logits.argmax())
    scaled = logits.float()
    scaled = (scaled - scaled.max()) / max(temperature, torch.finfo(torch.float32).tiny)
    if top_k and top_k < scaled.numel():
        threshold = torch.topk(scaled, top_k).values[-1]
        scaled = scaled.masked_fill(scaled < threshold, -torch.inf)
    if top_p < 1:
        sorted_logits, indices = scaled.sort(descending=True)
        probabilities = sorted_logits.softmax(dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        remove = cumulative - probabilities >= top_p
        scaled = scaled.scatter(0, indices, sorted_logits.masked_fill(remove, -torch.inf))
    return int(torch.multinomial(scaled.softmax(dim=-1), 1, generator=generator))


@torch.inference_mode()
def generate(model, input_ids, *, max_new_tokens, eos_token_id=None, generator=None, **sampling):
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or max_new_tokens < 0:
        raise ValueError("generation requires one prompt and a nonnegative output budget")
    if input_ids.shape[1] + max_new_tokens > model.config.max_seq_len:
        raise ValueError("prompt plus generation exceeds the trained context")
    if eos_token_id is not None and not 0 <= eos_token_id < model.config.vocab_size:
        raise ValueError("EOS is outside the vocabulary")
    if max_new_tokens == 0:
        return input_ids.clone()
    was_training = model.training
    model.eval()
    try:
        output, cache = model.forward_cached(input_ids, last_logits_only=True)
        generated = [input_ids]
        for index in range(max_new_tokens):
            token = sample_next_token(output["logits"][0, -1], generator=generator, **sampling)
            next_input = input_ids.new_tensor([[token]])
            generated.append(next_input)
            if token == eos_token_id or index + 1 == max_new_tokens:
                break
            output, cache = model.forward_cached(next_input, cache=cache, last_logits_only=True)
        return torch.cat(generated, dim=1)
    finally:
        model.train(was_training)
