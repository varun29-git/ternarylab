# Tokenizer Plan for 1.5B BitNet Multilingual Model
## English + 10 Indic Languages Training Project

---

## Executive Summary

This document outlines a comprehensive plan for creating a new multilingual tokenizer to support training a 1.5B parameter BitNet model on English and 10 Indic languages. The current TinyLlama tokenizer (32K vocab) is insufficient for multilingual training, particularly for Indic scripts with complex Unicode character compositions.

**Key Recommendations:**
- **Vocab Size:** 96,000 tokens (recommended middle ground)
- **Tokenizer Type:** SentencePiece Unigram
- **Training Corpus:** 10-15M sentences (5-10GB text)
- **Character Coverage:** 99.95% for all 11 languages
- **Special Tokens:** PAD, EOS, BOS, UNK, plus language-specific markers

---

## 1. Tokenizer Design Decisions

### 1.1 Vocab Size Selection

#### Analysis: 80K vs 96K vs 100K

| Vocab Size | Embedding Params | % of 1.5B Model | Pros | Cons |
|------------|------------------|-----------------|------|------|
| 80,000 | 163.8M (10.9%) | Lower memory footprint | May have suboptimal compression for Indic scripts |
| **96,000** | 196.6M (13.1%) | **Balanced coverage/efficiency** | Slightly larger embedding table |
| 100,000 | 204.8M (13.7%) | Maximum coverage | Diminishing returns, larger memory |

#### Recommendation: 96,000 tokens

**Rationale:**
1. **Indic Script Complexity:** Devanagari, Tamil, Telugu, Bengali, and other Indic scripts have complex character combinations (vowel signs, conjuncts) that benefit from larger vocabulary
2. **Compression Efficiency:** 96K provides ~4.2 characters per token for English (similar to GPT-4) while maintaining ~3.5 characters per token for Indic languages
3. **Parameter Budget:** At 196.6M parameters, embeddings represent 13.1% of total model - within optimal range of 10-15%
4. **Divisibility:** 96,000 is divisible by 8, 16, 32 - optimal for GPU tensor operations and distributed training

**Embedding Memory Calculation:**
```
Embedding: 96,000 × 2048 × 2 bytes (BF16) = 393.2 MB
Output head (tied): 0 bytes (shared with embedding)
Total embedding memory: ~393 MB
```

### 1.2 Tokenizer Type Selection

#### Comparison Matrix

| Tokenizer Type | Algorithm | Pros | Cons | Verdict |
|----------------|-----------|------|------|---------|
| **SentencePiece Unigram** | Probabilistic segmentation | Language-agnostic, handles morphologically rich languages, smooth vocab scaling | Slightly slower training | **RECOMMENDED** |
| SentencePiece BPE | Frequency-based merging | Widely used, deterministic | Less optimal for morphologically complex languages | Alternative |
| HuggingFace Tokenizers (BPE) | Byte-level BPE | Fast, well-integrated | Less flexible for custom preprocessing | Backup option |
| WordPiece | Greedy longest-match | Good for word-rich languages | Poor for agglutinative languages | Reject |

#### Recommendation: SentencePiece Unigram

**Rationale:**
1. **Morphological Handling:** Indic languages (especially Tamil, Telugu, Kannada) are morphologically rich - Unigram's probabilistic approach handles this better than BPE
2. **Proven Track Record:** Used by mT5, NLLB-200, and other successful multilingual models
3. **Vocab Flexibility:** Unigram allows smooth vocabulary size adjustment without retraining
4. **Subword Quality:** Produces more linguistically meaningful subwords for Indic scripts

### 1.3 Special Tokens

#### Required Special Tokens

| Token | ID | Purpose | Notes |
|-------|-----|---------|-------|
| `<pad>` | 0 | Padding | Required for batch processing |
| `<eos>` | 1 | End of sequence | Marks document/sentence boundaries |
| `<bos>` | 2 | Beginning of sequence | Optional but recommended for generation |
| `<unk>` | 3 | Unknown token | Fallback for OOV characters |
| `<mask>` | 4 | Masking token | For future MLM capabilities |

#### Language-Specific Tokens (Optional but Recommended)

| Token | ID Range | Purpose |
|-------|----------|---------|
| `<en>` | 5 | English language marker |
| `<hi>` | 6 | Hindi language marker |
| `<ta>` | 7 | Tamil language marker |
| `<te>` | 8 | Telugu language marker |
| `<bn>` | 9 | Bengali language marker |
| `<mr>` | 10 | Marathi language marker |
| `<gu>` | 11 | Gujarati language marker |
| `<kn>` | 12 | Kannada language marker |
| `<ml>` | 13 | Malayalam language marker |
| `<or>` | 14 | Odia language marker |
| `<pa>` | 15 | Punjabi language marker |

**Total special tokens:** 16 (IDs 0-15)
**Available for vocabulary:** 95,984 tokens (IDs 16-95,999)

### 1.4 Character Coverage Requirements

#### Indic Script Unicode Ranges

| Language | Script | Unicode Range | Characters | Coverage Target |
|----------|--------|---------------|------------|-----------------|
| Hindi | Devanagari | U+0900-U+097F | ~128 | 99.95% |
| Marathi | Devanagari | U+0900-U+097F | ~128 | 99.95% |
| Bengali | Bengali | U+0980-U+09FF | ~128 | 99.95% |
| Gujarati | Gujarati | U+0A80-U+0AFF | ~128 | 99.95% |
| Punjabi | Gurmukhi | U+0A00-U+0A7F | ~128 | 99.95% |
| Odia | Odia | U+0B00-U+0B7F | ~128 | 99.95% |
| Tamil | Tamil | U+0B80-U+0BFF | ~128 | 99.95% |
| Telugu | Telugu | U+0C00-U+0C7F | ~128 | 99.95% |
| Kannada | Kannada | U+0C80-U+0CFF | ~128 | 99.95% |
| Malayalam | Malayalam | U+0D00-U+0D7F | ~128 | 99.95% |
| English | Latin | U+0000-U+007F | ~128 | 99.99% |

**Total unique Unicode blocks:** 11 scripts
**Recommended character coverage setting:** 0.9995 (99.95%)

### 1.5 Code-Switching Strategy

#### Handling Mixed-Language Text

**Challenge:** Natural multilingual text often contains code-switching (e.g., Hindi-English mixed sentences like "Maine kal movie dekhi")

**Strategy:**
1. **No Script Boundary Tokens:** Let the tokenizer naturally handle script transitions
2. **Language Proportion Awareness:** Ensure training corpus includes realistic code-switching patterns
3. **Whitespace Preservation:** Maintain whitespace as natural boundary markers

**Training Corpus Code-Switching Proportions:**

| Scenario | Proportion | Example |
|----------|------------|---------|
| Monolingual English | 40% | "The weather is nice today." |
| Monolingual Indic | 35% | "आज मौसम अच्छा है।" |
| Indic-English mixed | 20% | "Maine kal movie dekhi, it was amazing!" |
| Multi-Indic mixed | 5% | "नमस्कार, আপনি কেমন আছেন?" |

---

## 2. Training Data Strategy

### 2.1 Corpus Composition

#### Recommended Training Corpus Size

| Metric | Minimum | Recommended | Maximum |
|--------|---------|-------------|---------|
| Sentences | 5M | 10-15M | 50M |
| Tokens (pre-tokenization) | 500M | 1-2B | 5B |
| Size (GB) | 2-3 GB | 5-10 GB | 20 GB |

**Recommendation:** 10-15M sentences (~5-10 GB of text)

**Rationale:**
- Too small: Vocabulary won't capture rare but important subwords
- Too large: Diminishing returns, longer training time
- Sweet spot: 10-15M sentences provides sufficient coverage for 96K vocab

### 2.2 Data Sources by Language

#### English Sources (50-60% of corpus)

| Source | Size | Content Type | Access |
|--------|------|--------------|--------|
| FineWeb-Edu sample | 2-3 GB | Educational web text | HuggingFace |
| Cosmopedia v2 sample | 2-3 GB | Synthetic educational | HuggingFace |
| Wikipedia English | 1 GB | Encyclopedia | HuggingFace |
| BookCorpus sample | 0.5 GB | Books | HuggingFace |
| **Total English** | **5-7 GB** | | |

#### Hindi Sources (15-20% of corpus)

| Source | Size | Content Type | Access |
|--------|------|--------------|--------|
| AI4Bharat Hindi corpus | 1-2 GB | News, Wikipedia, web | AI4Bharat |
| IndicNLP Hindi | 0.5-1 GB | News, literature | IndicNLP |
| Wikipedia Hindi | 0.3 GB | Encyclopedia | HuggingFace |
| Common Crawl Hindi | 0.5 GB | Web text | CC-100 |
| **Total Hindi** | **2-4 GB** | | |

#### Other Indic Languages (20-30% of corpus)

| Language | Primary Source | Secondary Source | Target Size |
|----------|----------------|------------------|-------------|
| Bengali | AI4Bharat Bengali | Wikipedia | 0.5-0.8 GB |
| Tamil | AI4Bharat Tamil | IndicNLP | 0.4-0.6 GB |
| Telugu | AI4Bharat Telugu | Wikipedia | 0.4-0.6 GB |
| Marathi | AI4Bharat Marathi | IndicNLP | 0.3-0.5 GB |
| Gujarati | AI4Bharat Gujarati | Wikipedia | 0.2-0.4 GB |
| Kannada | AI4Bharat Kannada | IndicNLP | 0.2-0.4 GB |
| Malayalam | AI4Bharat Malayalam | Wikipedia | 0.2-0.4 GB |
| Odia | AI4Bharat Odia | IndicNLP | 0.1-0.3 GB |
| Punjabi | AI4Bharat Punjabi | Wikipedia | 0.2-0.4 GB |
| **Total Other Indic** | | | **2.5-4.5 GB** |

### 2.3 Preprocessing Requirements

#### Text Normalization Pipeline

```python
def preprocess_text(text: str, language: str) -> str:
    """
    Normalization pipeline for tokenizer training.
    
    Steps:
    1. Unicode normalization (NFC)
    2. Whitespace normalization
    3. Control character removal
    4. Language-specific normalization
    """
    import unicodedata
    
    # Step 1: Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)
    
    # Step 2: Normalize whitespace
    text = ' '.join(text.split())
    
    # Step 3: Remove control characters (keep newlines for sentence boundaries)
    text = ''.join(
        char for char in text 
        if unicodedata.category(char) != 'Cc' or char == '\n'
    )
    
    # Step 4: Language-specific normalization
    if language in ['hi', 'mr', 'bn', 'gu', 'pa', 'or']:
        # Indic languages: Normalize visible vs invisible vowels
        # (Implementation depends on specific language requirements)
        pass
    
    return text.strip()
```

#### Cleaning Requirements

| Issue | Detection | Action |
|-------|-----------|--------|
| Excessive repetition | >3 consecutive identical tokens | Remove sentence |
| Too short | <5 characters | Remove sentence |
| Too long | >10000 characters | Split or remove |
| Mixed encoding | Non-UTF8 sequences | Remove or fix |
| Boilerplate | Common patterns (headers, footers) | Pattern-based removal |
| URLs/emails | Regex patterns | Replace with placeholders or remove |

#### Sentence Segmentation

**Recommended approach:** Language-aware sentence segmentation

```python
# Use nltk or specialized Indic segmenters
from indicnlp.tokenize.sentence_tokenize import sentence_split

def segment_sentences(text: str, language: str) -> List[str]:
    """Language-aware sentence segmentation."""
    if language == 'en':
        # Use nltk for English
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    else:
        # Use IndicNLP for Indic languages
        return sentence_split(text, lang=language)
```

### 2.4 Corpus Quality Metrics

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Perplexity (baseline) | <100 on held-out set | Train small LM on sample |
| Character distribution | Matches target language profiles | Unicode histogram analysis |
| Vocabulary coverage | >99.9% of characters | Character frequency analysis |
| Duplicate rate | <5% | Exact match detection |
| Average sentence length | 15-50 words | Statistical analysis |

---

## 3. Implementation Plan

### 3.1 Tools and Libraries

#### Primary Tool: SentencePiece

```bash
pip install sentencepiece>=0.2.0
```

**Why SentencePiece:**
- Native Unicode support for all Indic scripts
- Efficient memory usage for large corpora
- Supports both BPE and Unigram algorithms
- Well-documented and battle-tested

#### Supporting Libraries

```bash
pip install transformers>=4.40.0
pip install datasets>=2.18.0
pip install indicnlp>=0.0.5  # For Indic text processing
pip install nltk>=3.8.0      # For English sentence segmentation
```

### 3.2 Training Script

#### Main Training Script

```python
#!/usr/bin/env python3
"""
train_tokenizer.py
Train a multilingual SentencePiece tokenizer for English + 10 Indic languages.
"""

import argparse
import sentencepiece as spm
from pathlib import Path
from typing import List
import random

def create_training_corpus(
    input_files: List[Path],
    output_file: Path,
    sample_ratio: float = 1.0,
    seed: int = 42
) -> None:
    """
    Merge and sample input files into a single training corpus.
    
    Args:
        input_files: List of input text files
        output_file: Output file path
        sample_ratio: Fraction of data to sample (for large corpora)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            print(f"Processing {input_file}...")
            with open(input_file, 'r', encoding='utf-8') as in_f:
                lines = in_f.readlines()
                if sample_ratio < 1.0:
                    lines = random.sample(
                        lines, 
                        int(len(lines) * sample_ratio)
                    )
                out_f.writelines(lines)
    
    print(f"Created training corpus: {output_file}")
    print(f"Total lines: {sum(1 for _ in open(output_file, 'r'))}")

def train_tokenizer(
    input_file: Path,
    model_prefix: str,
    vocab_size: int = 96000,
    character_coverage: float = 0.9995,
    model_type: str = 'unigram',
    max_sentence_length: int = 8192,
    num_threads: int = 16,
    special_tokens: List[str] = None
) -> None:
    """
    Train SentencePiece tokenizer.
    
    Args:
        input_file: Training corpus file
        model_prefix: Output model prefix
        vocab_size: Target vocabulary size
        character_coverage: Character coverage threshold
        model_type: 'unigram' or 'bpe'
        max_sentence_length: Maximum sentence length in characters
        num_threads: Number of training threads
        special_tokens: List of special tokens to add
    """
    if special_tokens is None:
        special_tokens = [
            '<pad>', '<eos>', '<bos>', '<unk>', '<mask>',
            '<en>', '<hi>', '<ta>', '<te>', '<bn>',
            '<mr>', '<gu>', '<kn>', '<ml>', '<or>', '<pa>'
        ]
    
    # SentencePiece training arguments
    train_args = [
        f'--input={input_file}',
        f'--model_prefix={model_prefix}',
        f'--vocab_size={vocab_size}',
        f'--character_coverage={character_coverage}',
        f'--model_type={model_type}',
        f'--max_sentence_length={max_sentence_length}',
        f'--num_threads={num_threads}',
        '--shuffle_input_sentence=true',
        '--split_by_unicode_script=true',
        '--split_by_number=true',
        '--split_by_whitespace=true',
        '--split_digits=true',
        '--allow_whitespace_only_pieces=true',
        '--byte_fallback=true',  # Important for handling OOV characters
        f'--user_defined_symbols={",".join(special_tokens)}',
        '--pad_id=0',
        '--eos_id=1',
        '--bos_id=2',
        '--unk_id=3',
        '--pad_piece=<pad>',
        '--eos_piece=<eos>',
        '--bos_piece=<bos>',
        '--unk_piece=<unk>',
    ]
    
    print("Training tokenizer with arguments:")
    print(" ".join(train_args))
    
    spm.SentencePieceTrainer.train(" ".join(train_args))
    
    print(f"Tokenizer trained successfully: {model_prefix}.model")

def validate_tokenizer(
    model_path: Path,
    test_sentences: List[str]
) -> None:
    """
    Validate trained tokenizer on test sentences.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    
    print(f"\nVocabulary size: {sp.get_piece_size()}")
    print(f"Special tokens:")
    print(f"  PAD: {sp.pad_id()} -> '{sp.id_to_piece(sp.pad_id())}'")
    print(f"  EOS: {sp.eos_id()} -> '{sp.id_to_piece(sp.eos_id())}'")
    print(f"  BOS: {sp.bos_id()} -> '{sp.id_to_piece(sp.bos_id())}'")
    print(f"  UNK: {sp.unk_id()} -> '{sp.id_to_piece(sp.unk_id())}'")
    
    print("\nSample tokenizations:")
    for sentence in test_sentences:
        tokens = sp.encode(sentence, out_type=str)
        token_ids = sp.encode(sentence, out_type=int)
        print(f"  Input: {sentence[:50]}...")
        print(f"  Tokens: {tokens[:10]}...")
        print(f"  IDs: {token_ids[:10]}...")
        print()

def main():
    parser = argparse.ArgumentParser(
        description='Train multilingual tokenizer for Indic languages'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing training text files'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='multilingual_indic',
        help='Output model prefix'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=96000,
        help='Vocabulary size'
    )
    parser.add_argument(
        '--character-coverage',
        type=float,
        default=0.9995,
        help='Character coverage'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='unigram',
        choices=['unigram', 'bpe'],
        help='Tokenizer model type'
    )
    
    args = parser.parse_args()
    
    # Collect input files
    input_files = list(args.input_dir.glob('*.txt'))
    print(f"Found {len(input_files)} input files")
    
    # Create merged training corpus
    corpus_file = Path('training_corpus.txt')
    create_training_corpus(input_files, corpus_file)
    
    # Train tokenizer
    train_tokenizer(
        input_file=corpus_file,
        model_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type
    )
    
    # Validate tokenizer
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "आज मौसम बहुत अच्छा है।",
        "இன்று வானிலை மிகவும் நல்லது.",
        "ఈ రోజు వాతావరణం చాలా బాగుంది.",
        "Maine kal movie dekhi, it was amazing!",
    ]
    validate_tokenizer(
        Path(f"{args.output_prefix}.model"),
        test_sentences
    )

if __name__ == '__main__':
    main()
```

### 3.3 Hyperparameters

#### Recommended Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `vocab_size` | 96,000 | Balanced coverage/efficiency |
| `character_coverage` | 0.9995 | Covers 99.95% of characters |
| `model_type` | unigram | Better for morphologically rich languages |
| `max_sentence_length` | 8192 | Accommodate long documents |
| `split_by_unicode_script` | true | Handle script boundaries |
| `split_by_whitespace` | true | Preserve word boundaries |
| `split_digits` | true | Separate numbers |
| `byte_fallback` | true | Handle OOV characters gracefully |
| `shuffle_input_sentence` | true | Randomize training order |
| `num_threads` | 16 | Parallel training |

### 3.4 Validation and Quality Metrics

#### Compression Ratio Analysis

```python
def measure_compression_ratio(
    model_path: Path,
    test_files: List[Path]
) -> dict:
    """
    Measure compression ratio across test files.
    
    Returns:
        Dictionary with per-language compression metrics
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    
    results = {}
    
    for test_file in test_files:
        language = test_file.stem.split('_')[0]
        
        with open(test_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Character count
        char_count = len(text)
        
        # Token count
        tokens = sp.encode(text, out_type=int)
        token_count = len(tokens)
        
        # Compression ratio
        compression_ratio = char_count / token_count
        
        results[language] = {
            'characters': char_count,
            'tokens': token_count,
            'compression_ratio': compression_ratio
        }
        
        print(f"{language}: {compression_ratio:.2f} chars/token")
    
    return results
```

#### Target Compression Ratios

| Language | Target Compression | Notes |
|----------|-------------------|-------|
| English | 4.0-4.5 chars/token | Similar to GPT-4 |
| Hindi | 3.5-4.0 chars/token | Devanagari efficient encoding |
| Bengali | 3.2-3.8 chars/token | Bengali script |
| Tamil | 2.8-3.5 chars/token | Complex agglutinative |
| Telugu | 3.0-3.6 chars/token | Telugu script |
| Other Indic | 3.0-4.0 chars/token | Variable by script |

---

## 4. Integration Requirements

### 4.1 Code Changes in [`data.py`](archive/150m-30btokens-run-original/source/data.py)

#### Required Modifications

**1. Update `DEFAULT_MODEL_VOCAB_SIZE` (Line 20)**

```python
# Current
DEFAULT_MODEL_VOCAB_SIZE = 32_000

# Updated
DEFAULT_MODEL_VOCAB_SIZE = 96_000
```

**2. Update `_bounded_token_ids` function (Lines 613-634)**

```python
def _bounded_token_ids(
    ids: Sequence[int],
    *,
    eos_id: int,
    vocab_size: int = DEFAULT_MODEL_VOCAB_SIZE,  # Will use 96,000
) -> List[int]:
    """Keep streaming tokenization on the same vocab-size contract.
    
    The prepared memmaps were bounded before training. The streaming path
    tokenizes raw Hugging Face rows live, and some tokenizers can expose
    added/special IDs outside the model embedding table.
    Replace those rare IDs with EOS on CPU so CUDA never sees an illegal index.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    fallback = eos_id if 0 <= int(eos_id) < vocab_size else 0
    bounded: List[int] = []
    for token_id in ids:
        token = int(token_id)
        bounded.append(token if 0 <= token < vocab_size else fallback)
    return bounded
```

**3. Add tokenizer loading function**

```python
def load_multilingual_tokenizer(
    tokenizer_path: str | Path,
    tokenizer_type: str = "sentencepiece"
) -> Union[SentencePieceProcessor, "PreTrainedTokenizer"]:
    """
    Load multilingual tokenizer.
    
    Args:
        tokenizer_path: Path to tokenizer model
        tokenizer_type: 'sentencepiece' or 'huggingface'
    
    Returns:
        Tokenizer instance
    """
    if tokenizer_type == "sentencepiece":
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(tokenizer_path))
        return sp
    else:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_path)
```

**4. Update `StreamingTextTokenLoader` class**

Add support for multilingual tokenizer:

```python
class StreamingTextTokenLoader:
    def __init__(
        self,
        manifest: TokenManifest,
        *,
        tokenizer_path: str,  # Changed from tokenizer_name
        tokenizer_type: str = "sentencepiece",  # Added
        start_step: int = 0,
        max_steps: Optional[int] = None,
        prefetch_batches: int = 8,
        device: Optional[torch.device | str] = None,
        return_step: bool = False,
    ) -> None:
        # ... existing validation ...
        self.tokenizer_path = tokenizer_path
        self.tokenizer_type = tokenizer_type
        # ... rest of initialization ...
```

### 4.2 Config Updates

#### Model Configuration Updates

```json
{
  "vocab_size": 96000,
  "dim": 2048,
  "n_layers": 32,
  "n_heads": 16,
  "n_kv_heads": 8,
  "ffn_dim": 5504,
  "max_seq_len": 4096,
  "dtype": "bf16"
}
```

#### Train Configuration Updates

```json
{
  "tokenizer_path": "/path/to/multilingual_indic.model",
  "tokenizer_type": "sentencepiece",
  "vocab_size": 96000,
  "dim": 2048,
  "n_layers": 32,
  "n_heads": 16,
  "n_kv_heads": 8,
  "ffn_dim": 5504
}
```

### 4.3 Data Pipeline Retokenization Strategy

#### Option A: Full Retokenization (Recommended)

**Pros:**
- Clean slate, no legacy artifacts
- Optimal compression for new vocabulary
- Consistent tokenization across all data

**Cons:**
- Requires re-downloading/processing all source data
- Time-consuming (estimated 24-48 hours for 30B tokens)

**Implementation:**

```python
def retokenize_dataset(
    input_manifest: TokenManifest,
    output_manifest: TokenManifest,
    new_tokenizer_path: Path,
    output_dir: Path
) -> None:
    """
    Retokenize entire dataset with new tokenizer.
    
    Process:
    1. Load original text data
    2. Tokenize with new tokenizer
    3. Save as new binary memmap files
    4. Generate new manifest
    """
    import sentencepiece as spm
    import numpy as np
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(new_tokenizer_path))
    
    for shard in input_manifest.shards:
        # Load original data (would need access to raw text)
        # This assumes we have the raw text source
        # In practice, may need to re-download from HuggingFace
        
        # Tokenize
        # tokens = sp.encode(text, out_type=int)
        
        # Save as binary
        # np.array(tokens, dtype=np.uint32).tofile(output_path)
        
        pass
```

#### Option B: Hybrid Approach (If Raw Text Unavailable)

**Strategy:**
1. Keep existing English data tokenized with TinyLlama
2. Add new Indic data tokenized with new tokenizer
3. Use vocabulary mapping for compatibility

**Not recommended** - creates inconsistency in tokenization quality.

### 4.4 Backward Compatibility

#### Checkpoint Compatibility

**Issue:** Existing 150M model checkpoints use 32K vocabulary

**Solutions:**

| Approach | Description | Recommendation |
|----------|-------------|----------------|
| Fresh initialization | Start from scratch for 1.5B | **RECOMMENDED** - Clean slate |
| Vocabulary expansion | Add new tokens, keep old embeddings | Possible but suboptimal |
| Vocabulary mapping | Map old tokens to new tokenizer | Complex, not recommended |

**Recommendation:** Fresh initialization for 1.5B model

**Rationale:**
1. 1.5B model is 10× larger than 150M - weights won't transfer directly anyway
2. New vocabulary requires new embedding initialization
3. μP learning rate transfer (0.003) works without weight transfer

---

## 5. Evaluation Criteria

### 5.1 Compression Ratio Benchmarks

#### Benchmark Methodology

```python
def benchmark_compression(
    tokenizer_path: Path,
    test_data_dir: Path
) -> pd.DataFrame:
    """
    Benchmark tokenizer compression ratio across languages.
    
    Returns:
        DataFrame with per-language metrics
    """
    import pandas as pd
    import sentencepiece as spm
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
    
    results = []
    
    for test_file in test_data_dir.glob('*.txt'):
        language = test_file.stem
        
        with open(test_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Metrics
        char_count = len(text)
        tokens = sp.encode(text, out_type=int)
        token_count = len(tokens)
        
        # Unique tokens
        unique_tokens = len(set(tokens))
        
        # OOV rate (tokens that are byte fallbacks)
        byte_fallback_tokens = sum(
            1 for t in tokens 
            if sp.id_to_piece(t).startswith('▁<0x')
        )
        oov_rate = byte_fallback_tokens / token_count
        
        results.append({
            'language': language,
            'characters': char_count,
            'tokens': token_count,
            'compression_ratio': char_count / token_count,
            'unique_tokens': unique_tokens,
            'vocabulary_utilization': unique_tokens / sp.get_piece_size(),
            'oov_rate': oov_rate
        })
    
    return pd.DataFrame(results)
```

#### Target Benchmarks

| Language | Compression Ratio | OOV Rate | Vocab Utilization |
|----------|------------------|----------|-------------------|
| English | 4.0-4.5 | <0.01% | 60-70% |
| Hindi | 3.5-4.0 | <0.05% | 50-60% |
| Bengali | 3.2-3.8 | <0.05% | 40-50% |
| Tamil | 2.8-3.5 | <0.10% | 35-45% |
| Telugu | 3.0-3.6 | <0.10% | 35-45% |
| Marathi | 3.5-4.0 | <0.05% | 40-50% |
| Gujarati | 3.2-3.8 | <0.05% | 35-45% |
| Kannada | 3.0-3.6 | <0.10% | 35-45% |
| Malayalam | 2.8-3.5 | <0.10% | 35-45% |
| Odia | 3.0-3.6 | <0.10% | 30-40% |
| Punjabi | 3.2-3.8 | <0.05% | 35-45% |

### 5.2 Vocabulary Coverage Metrics

#### Character Coverage Analysis

```python
def analyze_character_coverage(
    tokenizer_path: Path,
    test_data_dir: Path
) -> dict:
    """
    Analyze character-level coverage for each script.
    """
    import sentencepiece as spm
    from collections import Counter
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
    
    # Get vocabulary characters
    vocab_chars = set()
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        # Remove sentencepiece prefix
        if piece.startswith('▁'):
            piece = piece[1:]
        vocab_chars.update(piece)
    
    results = {}
    
    for test_file in test_data_dir.glob('*.txt'):
        language = test_file.stem
        
        with open(test_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Count characters in test data
        test_chars = set(text)
        
        # Coverage
        covered = test_chars & vocab_chars
        uncovered = test_chars - vocab_chars
        
        results[language] = {
            'total_chars': len(test_chars),
            'covered_chars': len(covered),
            'coverage': len(covered) / len(test_chars) if test_chars else 1.0,
            'uncovered_chars': list(uncovered)[:20]  # First 20
        }
    
    return results
```

### 5.3 Per-Language Tokenization Quality

#### Quality Assessment Criteria

| Criterion | Measurement | Target |
|-----------|-------------|--------|
| Linguistic boundaries | % of tokens matching morpheme boundaries | >80% |
| Semantic coherence | Tokens should be meaningful units | Manual review |
| Rare word handling | OOV rate for rare words | <1% |
| Number handling | Digits tokenized appropriately | Visual inspection |
| Punctuation | Preserved as separate tokens | 100% |

### 5.4 Comparison with Existing Multilingual Tokenizers

#### Benchmark Against Reference Tokenizers

| Tokenizer | Vocab Size | English Compression | Hindi Compression | Notes |
|-----------|------------|---------------------|-------------------|-------|
| **Ours** | 96,000 | Target: 4.0-4.5 | Target: 3.5-4.0 | Optimized for 11 languages |
| mT5 | 250,000 | ~3.5 | ~3.0 | Too large, over-parameterized |
| NLLB-200 | 256,000 | ~3.8 | ~3.2 | Covers 200 languages |
| IndicBERT | 100,000 | ~3.5 | ~3.8 | Good for Indic, weak for English |
| XLM-R | 250,000 | ~3.5 | ~3.0 | Too large |
| GPT-4 | 100,000 | ~4.2 | N/A | English-only baseline |

#### Expected Performance

Our tokenizer should achieve:
- **Better English compression** than IndicBERT (due to larger English corpus)
- **Better Indic compression** than GPT-4 (which doesn't support Indic)
- **Comparable compression** to mT5 with 2.5× smaller vocabulary

---

## 6. Timeline and Resources

### 6.1 Phase Breakdown

#### Phase 1: Data Collection and Preparation

| Task | Dependencies | Deliverable |
|------|--------------|-------------|
| Download English corpora | None | 5-7 GB English text |
| Download Hindi corpora | None | 2-4 GB Hindi text |
| Download other Indic corpora | None | 2.5-4.5 GB Indic text |
| Text preprocessing | Data download | Cleaned corpus files |
| Quality validation | Preprocessing | Quality report |

#### Phase 2: Tokenizer Training

| Task | Dependencies | Deliverable |
|------|--------------|-------------{}