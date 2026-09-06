> **Historical / superseded:** This research or planning document is not the supported training specification. Its architectural, tokenizer, scaling and performance claims are not release guarantees. Use `README.md` and `reports/PRETRAINING_AUDIT.md` at the repository root for the maintained implementation, measured evidence and outstanding launch gates.

# Post-Training Alignment Research Report
## 1.5B Parameter Multilingual LLM (English + Indic Languages)

---

## Executive Summary

This report presents research findings on optimal post-training alignment strategies for a 1.5B parameter multilingual LLM targeting English and 10 Indic languages. Based on 2024-2025 research, we recommend **SimPO** as the primary preference optimization method due to its compute efficiency and superior performance, combined with a carefully curated SFT dataset mixing English and multilingual instruction data.

---

## 1. Preference Optimization Methods Comparison

### 1.1 DPO (Direct Preference Optimization)

| Aspect | Details |
|--------|---------|
| **Paper** | [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) |
| **Authors** | Archit Sharma et al. (2023, updated 2024) |
| **Pros** | Stable, performant, computationally lightweight; eliminates need for sampling during fine-tuning; no significant hyperparameter tuning required |
| **Cons** | Requires reference model (memory overhead); can be sensitive to data quality |
| **Compute Cost** | Medium (requires reference model, ~2x base model memory) |
| **Multilingual Support** | Good - language-agnostic loss function |

### 1.2 SimPO (Simple Preference Optimization)

| Aspect | Details |
|--------|---------|
| **Paper** | [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) |
| **Authors** | Yu Meng et al. (NeurIPS 2024) |
| **Pros** | No reference model needed; more compute and memory efficient; outperforms DPO by up to 6.4 points on AlpacaEval 2; introduces target reward margin |
| **Cons** | Newer method, less established track record |
| **Compute Cost** | Low (no reference model) |
| **Multilingual Support** | Excellent - uses average log probability of sequence |

### 1.3 ORPO (Odds Ratio Preference Optimization)

| Aspect | Details |
|--------|---------|
| **Paper** | [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) |
| **Authors** | Jiwoo Hong et al. (2024) |
| **Pros** | Reference-free; monolithic approach; works well from 125M to 7B parameters; achieves 12.20% on AlpacaEval 2.0 |
| **Cons** | Requires SFT as prerequisite; newer method |
| **Compute Cost** | Low to Medium |
| **Multilingual Support** | Good |

### 1.4 PPO-based RLHF (InstructGPT approach)

| Aspect | Details |
|--------|---------|
| **Paper** | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) |
| **Authors** | Long Ouyang et al. (OpenAI, 2022) |
| **Pros** | Proven at scale; 1.3B InstructGPT preferred over 175B GPT-3 |
| **Cons** | Complex, often unstable; requires fitting reward model first; high compute requirements |
| **Compute Cost** | Very High (reward model + PPO training) |
| **Multilingual Support** | Requires language-specific reward data |

### Comparison Table

| Method | Pros | Cons | Compute Cost | Multilingual Support |
|--------|------|------|--------------|---------------------|
| **DPO** | Stable, well-established | Reference model needed | Medium | Excellent |
| **SimPO** | No reference needed, best performance | Newer method | Low | Excellent |
| **ORPO** | Works at small scales | Requires SFT first | Low-Medium | Good |
| **PPO/RLHF** | Proven at scale | Complex, unstable, expensive | Very High | Requires specific data |

---

## 2. SFT Dataset Recommendations

### 2.1 English Instruction Datasets

| Dataset | Description | Size | Source |
|---------|-------------|------|--------|
| **Alpaca** | Instruction following | 52K | [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) |
| **FLAN** | Collection of NLP tasks | 1.8M+ | [Google FLAN](https://github.com/google-research/FLAN) |
| **OpenOrca** | Reasoning explanations | 1M+ | [OpenOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca) |
| **UltraChat** | Multi-turn dialogue | 1.4M | [UltraChat](https://github.com/thunlp/UltraChat) |

### 2.2 Multilingual/Indic Instruction Datasets

| Dataset | Languages | Source |
|---------|-----------|--------|
| **AI4Bharat-Instruct** | Hindi, Tamil, Telugu, Bengali, + | [AI4Bharat](https://ai4bharat.org/) |
| **Indic-Instruct** | 10 Indic languages | [HuggingFace](https://huggingface.co/datasets?search=indic+instruct) |
| **Multilingual Alpaca** | 20+ languages | [Multilingual Alpaca](https://github.com/OptimalScale/FlanAlpaca) |
| **Bactrian-X** | 50+ languages | [Bactrian-X](https://huggingface.co/datasets/MiniMaxAI/Bactrian-X) |

### 2.3 Recommended Mixing Ratios

For a 1.5B model with 11 languages:

- **English SFT data**: 60-70% (foundation capability)
- **Indic language SFT data**: 30-40% (language-specific skills)
  - Hindi: 10-15% (largest Indic language)
  - Tamil/Telugu/Bengali: 3-5% each
  - Other 7 languages: 1-2% each

**Total SFT tokens estimate**: 50-100K examples (approximately 100M-200M tokens)

---

## 3. Preference Optimization Recommendation

### 3.1 Primary Recommendation: SimPO

**Rationale for 1.5B multilingual model:**

1. **Compute Efficiency**: No reference model needed - critical for 8,000 GPU-hour budget
2. **Performance**: Outperforms DPO by up to 6.4% on AlpacaEval 2
3. **Scalability**: Works well across model sizes (tested on Gemma-2-9B, Llama 3, Mistral)
4. **Multilingual**: Language-agnostic reward formulation

### 3.2 Implementation Complexity

| Phase | Complexity | Estimated GPU Hours |
|-------|------------|---------------------|
| SFT | Medium | 2,000-3,000 |
| Preference Data Collection | High | 500-1,000 |
| SimPO Training | Low | 1,000-2,000 |
| Evaluation | Medium | 500-1,000 |
| **Total** | - | **4,000-7,000** |

### 3.3 Compute Budget Allocation

Within 8,000 GPU-hour budget:
- **SFT Phase**: 2,500 GPU-hours (31%)
- **Preference Data Curation**: 750 GPU-hours (9%)
- **SimPO Training**: 1,500 GPU-hours (19%)
- **Evaluation & Iteration**: 1,250 GPU-hours (16%)
- **Contingency**: 2,000 GPU-hours (25%)

---

## 4. Evaluation Benchmarks

### 4.1 General Reasoning Benchmarks (1.5B scale appropriate)

| Benchmark | Description |适用性 |
|-----------|-------------|-------|
| **MMLU** | Multi-task language understanding | Good |
| **HellaSwag** | Commonsense reasoning | Good |
| **ARC** | Abstraction and reasoning | Good |
| **TruthfulQA** | Truthfulness evaluation | Good |
| **BBH** (Big-Bench Hard) | Challenging reasoning | Limited (too hard) |

### 4.2 Indic Language Benchmarks

| Benchmark | Languages | Source |
|-----------|-----------|--------|
| **IndicGLUE** | 10 Indic languages | [IndicGLUE](https://indicnlp.ai4bharat.org/indic-glue/) |
| **IndicXTREME** | 10 Indic languages | [IndicXTREME](https://github.com/AI4Bharat/IndicXTREME) |
| **IndicBERT** | 10 Indic languages | [AI4Bharat](https://ai4bharat.org/indic-bert) |
| **MASSIVE** | 50 languages (incl. Hindi) | [MASSIVE](https://github.com/alexa/massive) |

### 4.3 Chat/Assistant Benchmarks

| Benchmark | Description |
|-----------|-------------|
| **MT-Bench** | Multi-turn dialogue evaluation |
| **AlpacaEval** | Instruction following evaluation |
| **ChatArena** | Human preference evaluation |
| **IFEval** | Instruction-level evaluation |

---

## 5. Multilingual Alignment Challenges

### 5.1 Language-Specific Preference Data

- **Challenge**: Limited preference data for Indic languages
- **Solution**: Use English preference data for initial alignment, then adapt with smaller language-specific datasets
- **Transfer learning**: Research shows cross-lingual transfer works well for preference learning

### 5.2 Cross-Lingual Alignment Transfer

- English preference data can transfer to other languages due to shared reasoning patterns
- Fine-tune on mixed-language preference data for best results
- Consider language-specific safety guidelines

### 5.3 Safety Alignment for Indic Languages

- Limited safety datasets for Indic languages
- Recommend: Start with English safety data, then expand
- Key areas: Hate speech detection, harmful content filtering, cultural sensitivity

---

## 6. Source Citations

### Primary Papers

1. **DPO**: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
   - URL: https://arxiv.org/abs/2305.18290
   - Authors: Archit Sharma et al.
   - Published: May 2023, revised July 2024

2. **SimPO**: "SimPO: Simple Preference Optimization with a Reference-Free Reward"
   - URL: https://arxiv.org/abs/2405.14734
   - Authors: Yu Meng et al.
   - Published: May 2024 (NeurIPS 2024)

3. **ORPO**: "ORPO: Monolithic Preference Optimization without Reference Model"
   - URL: https://arxiv.org/abs/2403.07691
   - Authors: Jiwoo Hong et al.
   - Published: March 2024

4. **InstructGPT**: "Training language models to follow instructions with human feedback"
   - URL: https://arxiv.org/abs/2203.02155
   - Authors: Long Ouyang et al. (OpenAI)
   - Published: March 2022

5. **Early RLHF**: "Fine-Tuning Language Models from Human Preferences"
   - URL: https://arxiv.org/abs/1909.08593
   - Authors: Daniel M. Ziegler et al.
   - Published: September 2019

### Benchmark Sources

6. **IndicGLUE**: https://indicnlp.ai4bharat.org/indic-glue/
7. **IndicXTREME**: https://github.com/AI4Bharat/IndicXTREME
8. **AI4Bharat**: https://ai4bharat.org/

---

## 7. Recommendations Summary

### For 1.5B Multilingual Model Alignment:

1. **SFT Phase**: 
   - Use 60% English + 40% Indic instruction data
   - Target: 50-100K examples

2. **Preference Optimization**:
   - **Primary**: SimPO (compute efficient, best performance)
   - **Alternative**: DPO (more established)

3. **Compute Budget** (8,000 GPU-hours):
   - SFT: 2,500 hours
   - SimPO: 1,500 hours
   - Evaluation: 1,250 hours
   - Data & contingency: 2,750 hours

4. **Evaluation**:
   - General: MMLU, HellaSwag, TruthfulQA
   - Indic: IndicGLUE, IndicXTREME
   - Chat: MT-Bench, AlpacaEval

---

*Report generated: August 2026*
*Research focus: 2024-2025 alignment methods*
