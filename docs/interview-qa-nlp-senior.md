# Interview Q&A — NLP (senior / staff level)

*Role-agnostic themes for **AI/ML interviews** where **language** matters: LLMs, retrieval, evaluation, production, and risk. Use alongside project-specific docs (e.g. enterprise RAG, MLOps).*

**How to use these answers:** Each **Answer** starts with an **easy-to-follow core**; **If they want more:** adds names, metrics, and senior follow-through—same pattern as [`interview-qa-rag-senior.md`](interview-qa-rag-senior.md).

**Interview tip:** Open with a **clear sentence**, then **tradeoffs**, **failure modes**, and **what you’d measure**—not jargon first.

---

## Representations, tokenization, and “meaning”

### 1. Why do modern NLP systems use subword tokenization (BPE, SentencePiece) instead of words?

**Answer:** Real text contains **typos, rare words, and new words** you cannot list in advance. **Subword** tokenizers split words into **smaller pieces** so the model still has ids for rare inputs and does not need a **huge** fixed vocabulary.

**If they want more:** Tradeoff: the same word can split differently after a typo—**normalization** and **robust preprocessing** matter in production (BPE, SentencePiece, etc.).

---

### 2. Static embeddings (Word2Vec, GloVe) vs contextual embeddings—what changes in a senior explanation?

**Answer:** **Old-style** embeddings give **one vector per word** everywhere—“bank” is the same in “river bank” and “bank account.” **Contextual** models (transformers) build a vector from the **whole sentence**, so the same word can mean different things in different places.

**If they want more:** Static embeddings are still OK for **lightweight** search baselines; contextual models win when **meaning in context** drives quality.

---

### 3. What is the practical downside of relying on embedding cosine similarity as “semantic truth”?

**Answer:** Similarity scores are **not probabilities**—“0.7 similar” is **not** “70% correct.” Embeddings can favor **long** or **common** phrases and **drift** when the domain differs from training data.

**If they want more:** Mitigations: **reranking**, **keyword + vector hybrid**, **human or LLM judges** on a sample, **calibrated** thresholds from eval—not raw cosine alone.

---

## Transformers, architectures, and model families

### 4. Encoder-only vs decoder-only vs encoder–decoder—when would you choose each?

**Answer:** **Encoder-only** (BERT-style): read the whole input at once—great for **classification** and **embeddings**. **Decoder-only** (GPT-style): generate text **one token at a time**—what most chat LLMs use. **Encoder–decoder**: classic **translation / summarization** with separate encoder and decoder—many products now use **decoder-only + prompts** instead.

**If they want more:** Pick by **task**, **latency**, and **what your vendor/API** supports.

---

### 5. What does “self-attention” buy you that RNNs struggled with?

**Answer:** Each token can **look at every other token in one step** (within a layer), so long-range dependencies are easier than in old **recurrent** models that passed signal step-by-step.

**If they want more:** Cost grows roughly with **sequence length squared**, which is why models use **long-context tricks** (sparse attention, KV cache tuning, etc.).

---

### 6. How would you explain KV cache to a non-research interviewer in one minute?

**Answer:** When the model generates the **next** word, it should not **recompute** everything it already computed for **previous** words. The **KV cache** stores those past results so each new token is **cheap**—huge for **latency and cost** at long prompts.

**If they want more:** Context length, batch size, and **precision** (FP16/BF16/INT8) drive **serving economics**.

---

## Training paradigms: pretrain, fine-tune, align, adapt

### 7. Pre-training vs fine-tuning vs prompt engineering—how do you decide?

**Answer:** **Prompting** is fastest: try clear instructions first. **Fine-tuning** when the model **repeatedly** fails at format, tone, or domain wording prompts cannot fix. **Training from scratch** is rare for most teams—you **start from a foundation model**.

**If they want more:** Never fine-tune without a **real eval set** and a **rollback** plan; align spend with **data you actually have**.

---

### 8. What is PEFT / LoRA, and why does it matter in production?

**Answer:** Instead of updating **all billions** of weights, you train a **small adapter** (e.g. low-rank matrices) on top of a frozen base model—**less GPU memory**, **faster** iterations, **many specialized variants** of one base.

**If they want more:** Tradeoff: **less capacity** than full fine-tune; **eval** shows whether adapters are enough.

---

### 9. What is RLHF / preference optimization trying to fix that supervised fine-tuning does not?

**Answer:** **Supervised** learning copies **example answers**. **Preferences** are often “A is better than B”—RLHF / DPO-style methods optimize toward **what humans prefer** (helpful, harmless), not just imitation.

**If they want more:** These methods add **instability** and **eval complexity**—treat as **high-risk** changes: offline eval, **canaries**, **rollback**.

---

## Retrieval, grounding, and RAG (ties to your projects)

### 10. When is RAG strictly better than “fine-tune the facts in”?

**Answer:** When facts **change often** (policies, prices), must be **quoted from sources**, or are **too large** to memorize in weights. RAG **updates the document index**; fine-tuning **bakes facts into weights** that go **stale**.

**If they want more:** RAG adds **search + infra** and **retrieval failures**—not free.

---

### 11. What failure modes does RAG add that plain LLM inference does not?

**Answer:** You can retrieve **wrong or noisy chunks**, get **no hits**, **duplicate** chunks, **stale** index, **bad chunk boundaries**, or **malicious text** in documents that steers the model (**prompt injection** via corpus).

**If they want more:** Mitigations: **hybrid search**, **reranking**, **thresholds**, **“no answer”** paths, **guardrails**, **metrics** on retrieval.

---

### 12. What does “grounding” mean in product language, and how do you test it?

**Answer:** **Grounding** means the answer is **actually supported by** the evidence you showed the model—not just fluent. You test with **human review**, **citation checks**, and **automated** helpers (e.g. entailment-style checks)—knowing **no single score** is perfect.

---

## Evaluation: what “good” means for NLP systems

### 13. Why are BLEU/ROUGE insufficient for modern LLM quality?

**Answer:** They mostly score **word overlap** with one **reference** answer. Many **correct** answers use different words, so overlap **punishes** good paraphrases; they also ignore **correctness**, **helpfulness**, and **safety**.

**If they want more:** Use **task rubrics**, **human** labels, and **judges** where appropriate—BLEU/ROUGE are narrow tools, not **assistant** quality.

---

### 14. How would you design a minimal eval harness for a domain Q&A assistant?

**Answer:** A **fixed set** of real questions with **expected facts or citations**; **automatic** checks (did retrieval hit the right doc? JSON shape?); **latency and cost** tracked; optional **LLM judge** only if **calibrated** to humans. **Slice** results by topic and user type—**averages hide** bad pockets.

---

### 15. What is “data contamination” in eval, and why is it a senior concern?

**Answer:** If **test questions** (or answers) appeared in **training data** or **leaked prompts**, scores look **too good** and **miscompare** models. Senior teams use **private held-out sets**, **dynamic** questions, and treat public benchmarks as **hints**, not gospel.

---

## Hallucination, safety, bias, and governance

### 16. Define hallucination in a way you can defend in a review.

**Answer:** The model says something **confidently** that is **not justified** by the **inputs** you gave it or **trusted** sources—**invented** facts, **wrong numbers**, **fake citations**.

**If they want more:** It is not only randomness; **pressure** to always answer and **thin context** make it **systematic**.

---

### 17. What mitigations reduce hallucination in production systems?

**Answer:** **Ground with retrieval or tools**, **refuse** when evidence is weak, **lower temperature** where appropriate, **structured outputs**, **human review** for high risk, and **monitor** known bad patterns.

**If they want more:** No single fix—combine **product rules** (what the bot may say) with **engineering**.

---

### 18. How do you talk about bias in NLP systems without hand-waving?

**Answer:** Name **who gets hurt** and **how**: unfair **stereotypes**, uneven **quality** across dialects or demographics, unfair **decisions**. Mitigations: **audit data**, **measure by subgroup**, **policies**, and **human oversight**—with **honest limits**, not one “fairness number.”

---

## Production: latency, cost, multilingual

### 19. What are the main latency drivers at inference for LLM-heavy NLP?

**Answer:** **Time to process the prompt** (prefill), **time to generate each new token** (decode), **model size**, **network**, plus **RAG/tool** steps. Users care about **tail latency** (p95) and **cost per successful task**, not only averages.

---

### 20. How do you approach multilingual NLP when the product is “English-first”?

**Answer:** Do **not** assume **equal** quality in every language. **Measure** per language; consider **translate-then-answer**, **language-specific** prompts, **retrieval** in the user’s language, and **human review** where regulated.

---

### 21. What is “catastrophic forgetting” in NLP adaptation, and when does it matter?

**Answer:** After **new** fine-tuning, the model may get **worse** at **old** tasks it used to do well—new updates **overwrite** prior behavior.

**If they want more:** Mitigations: **mix** old and new data, **regularization**, **LoRA** with routing, **eval on both** old and new tasks.

---

## Synthesis: senior “story” questions

### 22. How would you explain the last 10 years of NLP progress in three sentences?

**Answer:** Deep learning replaced **hand-built features** with **learned representations**. **Transformers** scaled **self-supervised** training on huge text. **Instruction tuning and alignment** turned base models into **assistants**; **RAG and tools** connect them to **fresh data** and **actions**—with real **cost, latency, and reliability** tradeoffs.

---

### 23. If stakeholders ask for “100% accuracy,” how do you respond?

**Answer:** For open-ended language, **100%** is rarely meaningful without defining **exactly** what “right” means. Narrow the task (**structured extraction** with validation), add **human escalation**, and agree on **measurable** error rates and **monitoring**—not a magic guarantee.

---

### 24. What separates a “demo NLP feature” from a “production NLP system”?

**Answer:** Production has **eval** (offline and live), **monitoring**, **fallbacks**, **security** (e.g. injection, data leaks), **governance** (PII, retention), and **people** who can **on-call** and **roll back**. Demos optimize **happy paths**; production lives in **edge cases**.

---

## One-page cheat sheet


| Theme            | Senior sound bite                                                        |
| ---------------- | ------------------------------------------------------------------------ |
| **Tokenization** | Subwords **reduce OOV**; watch **tokenization fragility**.               |
| **Contextual**   | Meaning is **context-dependent**; static embeddings are **limited**.     |
| **Architecture** | Encoder vs decoder vs seq2seq maps to **task + infra reality**.          |
| **Adaptation**   | **PEFT** for speed; **RLHF** for alignment—both need **rigorous eval**.  |
| **RAG**          | **Fresh knowledge** + **citations** with **retrieval failure modes**.    |
| **Eval**         | **Overlap metrics** are not enough; **slice** and **calibrate judges**.  |
| **Risk**         | **Hallucination** and **bias** need **systems**, not just prompts.       |
| **Prod**         | **P95/P99**, **cost**, **KV cache**, **multilingual**—not just accuracy. |

