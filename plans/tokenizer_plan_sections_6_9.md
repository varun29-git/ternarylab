# Tokenizer Plan - Sections 6-9 (Continuation)

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
|------|--------------|-------------|
| Initial training run | Phase 1 complete | Draft tokenizer model |
| Hyperparameter tuning | Initial training | Optimized config |
| Validation testing | Training complete | Quality metrics report |
| Final model selection | Validation | Production tokenizer |

#### Phase 3: Integration

| Task | Dependencies | Deliverable |
|------|--------------|-------------|
| Update data.py | Tokenizer final | Modified source code |
| Update config files | Tokenizer final | New configs |
| Retokenize training data | Integration code | New binary memmap files |
| Integration testing | Retokenization | Test results |

#### Phase 4: Validation

| Task | Dependencies | Deliverable |
|------|--------------|-------------|
| Compression benchmarks | Integration complete | Benchmark report |
| Coverage analysis | Integration complete | Coverage report |
| Comparison with baselines | Benchmarks | Comparison report |
| Final documentation | All phases | Complete plan doc |

### 6.2 Computational Requirements

#### Tokenizer Training Resources

| Resource | Requirement | Notes |
|----------|-------------|-------|
| CPU cores | 16-32 | SentencePiece is CPU-bound |
| RAM | 32-64 GB | Large corpus in memory |
| Disk | 100 GB | Corpus + intermediate files |
| GPU | None required | Tokenizer training is CPU-only |
| Training time | 4-12 hours | Depends on corpus size |

#### Retokenization Resources

| Resource | Requirement | Notes |
|----------|-------------|-------|
| CPU cores | 8-16 | Parallel processing |
| RAM | 64-128 GB | Large memmap files |
| Disk | 500 GB | Original + retokenized data |
| GPU | Optional | For validation |
| Processing time | 24-48 hours | For 30B tokens |

### 6.3 Dependencies and Blockers

#### Critical Dependencies

| Dependency | Type | Status | Mitigation |
|------------|------|--------|------------|
| AI4Bharat corpus access | Data | Available | Download via HuggingFace |
| IndicNLP corpus access | Data | Available | Download via official site |
| SentencePiece library | Software | Stable | pip install |
| HuggingFace datasets | Software | Stable | pip install |
| Compute resources | Hardware | Required | Local or cloud |

#### Potential Blockers

| Blocker | Risk Level | Mitigation |
|---------|------------|------------|
| Insufficient Indic text data | Medium | Use augmentation, synthetic data |
| Poor compression for some languages | Medium | Adjust corpus proportions |
| Unicode normalization issues | Low | Test thoroughly, use NFC |
| Memory constraints during training | Low | Use streaming, reduce corpus |

### 6.4 Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Vocabulary size | 96,000 ± 1,000 | SentencePiece output |
| Character coverage | >99.95% | Coverage analysis |
| English compression | 4.0-4.5 chars/token | Benchmark test |
| Hindi compression | 3.5-4.0 chars/token | Benchmark test |
| Other Indic compression | 3.0-4.0 chars/token | Benchmark test |
| OOV rate | <0.1% | Coverage analysis |
| Training time | <12 hours | Wall clock |
| Integration success | All tests pass | Unit tests |

---

## 7. Risk Assessment

### 7.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Poor Indic script coverage | Medium | High | Increase character_coverage, add more Indic corpus |
| Vocabulary too large | Low | Medium | Reduce vocab_size, retrain |
| Vocabulary too small | Medium | High | Increase vocab_size, retrain |
| Byte fallback overuse | Medium | Medium | Increase corpus for affected languages |
| Training instability | Low | Low | Use stable SentencePiece version |

### 7.2 Data Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Insufficient Hindi data | Low | High | Use data augmentation |
| Noisy corpus | Medium | Medium | Better preprocessing |
| Imbalanced language distribution | Medium | Medium | Adjust corpus proportions |
| Copyright/licensing issues | Low | High | Use only open datasets |

### 7.3 Integration Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking existing pipeline | Medium | High | Thorough testing, staged rollout |
| Performance regression | Low | Medium | Benchmark before/after |
| Memory increase | High | Low | Expected with larger vocab |
| Checkpoint incompatibility | High | Medium | Fresh initialization planned |

---

## 8. Appendix

### 8.1 Indic Script Unicode Reference

```
Devanagari (Hindi, Marathi):    U+0900-U+097F
Bengali:                        U+0980-U+09FF
Gurmukhi (Punjabi):             U+0A00-U+0A7F
Gujarati:                       U+0A80-U+0AFF
Odia:                           U+0B00-U+0B7F
Tamil:                          U+0B80-U+0BFF
Telugu:                         U+0C00-U+0C7F
Kannada:                        U+0C80-U+0CFF
Malayalam:                      U+0D00-U+0D7F
```

### 8.2 Sample Tokenization Examples

#### English
```
Input:  "The quick brown fox jumps over the lazy dog."
Tokens: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
IDs:    [3847, 15234, 8923, 4521, 23456, 892, 312, 17892, 4567, 26]
```

#### Hindi (Devanagari)
```
Input:  "आज मौसम बहुत अच्छा है।"
Tokens: ['आज', 'मौसम', 'बहुत', 'अच्छा', 'है', '।']
IDs:    [12456, 23456, 17892, 34567, 12345, 789]
```

#### Tamil
```
Input:  "இன்று வானிலை மிகவும் நல்லது."
Tokens: ['இன்று', 'வானிலை', 'மிகவும்', 'நல்லது', '.']
IDs:    [45678, 56789, 45612, 34567, 26]
```

#### Code-Switching (Hindi-English)
```
Input:  "Maine kal movie dekhi, it was amazing!"
Tokens: ['Maine', 'kal', 'movie', 'dekhi', ',', 'it', 'was', 'amazing', '!']
IDs:    [23456, 17892, 45678, 23456, 25, 892, 456, 12345, 27]
```

### 8.3 Configuration File Templates

#### tokenizer_config.json
```json
{
  "tokenizer_type": "sentencepiece",
  "vocab_size": 96000,
  "model_type": "unigram",
  "character_coverage": 0.9995,
  "special_tokens": [
    "<pad>", "<eos>", "<bos>", "<unk>", "<mask>",
    "<en>", "<hi>", "<ta>", "<te>", "<bn>",
    "<mr>", "<gu>", "<kn>", "<ml>", "<or>", "<pa>"
  ],
  "languages": {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "or": "Odia",
    "pa": "Punjabi"
  }
}
```

### 8.4 References

1. **SentencePiece Documentation**
   - https://github.com/google/sentencepiece

2. **AI4Bharat Resources**
   - https://ai4bharat.org/
   - https://huggingface.co/ai4bharat

3. **IndicNLP Library**
   - https://github.com/anoopkunchukuttan/indic_nlp_library

4. **Multilingual Tokenization Research**
   - Tokenization and the Noiseless Channel (Rust et al., 2021)
   - How Good is Your Tokenizer? (Ali et al., 2024)

5. **Existing Multilingual Tokenizers**
   - mT5: https://github.com/google/mt5
   - NLLB: https://github.com/facebookresearch/fairseq/tree/nllb
   - IndicBERT: https://github.com/AI4Bharat/indic-bert

---

## 9. Summary

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vocab size | **96,000** | Balanced coverage for 11 languages |
| Tokenizer type | **SentencePiece Unigram** | Best for morphologically rich Indic languages |
| Character coverage | **99.95%** | Ensures all Indic scripts covered |
| Training corpus | **10-15M sentences** | Sufficient for 96K vocab |
| Special tokens | **16 tokens** | PAD, EOS, BOS, UNK, MASK + 11 language markers |

### Implementation Priority

1. **High Priority:** Data collection, tokenizer training, validation
2. **Medium Priority:** Integration, retokenization, testing
3. **Low Priority:** Documentation, optimization

### Next Steps

1. Begin data collection from AI4Bharat and IndicNLP sources
2. Set up preprocessing pipeline for 11 languages
3. Train initial tokenizer with recommended configuration
4. Validate compression ratios and coverage metrics
5. Integrate with training pipeline once validated

---

*Document generated: 2026-08-28*
*Version: 1.0*
*Status: Complete*
