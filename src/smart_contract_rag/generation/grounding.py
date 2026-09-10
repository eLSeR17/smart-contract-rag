"""Anti-hallucination grounding: can the answer's claims be verified against the sources?

Grounding is the output-side safety net of the RAG pipeline. Even if the LLM is
instructed to cite sources, it can still drift. ``GroundedTextCheck`` verifies
that the answer is supported by the retrieved chunks; if it is not, the
response is deemed ungrounded and the pipeline refuses it.

Two implementations are provided:

    * :class:`NaiveGroundedTextCheck` (default) — a *lexical* checker. It
      tests whether the *substantive tokens* of the answer (numbers,
      identifiers, technical terms) literally appear in the retrieved chunks.
      Deterministic, CI-safe and cheap, but it rejects correctly *synthesised*
      answers whose phrasing differs from the source (a conclusion that is a
      logical consequence of the evidence may not share its vocabulary).

    * :class:`EntailmentGroundedTextCheck` (experimental) — a *semantic*
      checker. It uses a local NLI cross-encoder to score whether the answer is
      *entailed* by each retrieved chunk, accepting conclusions that are
      logically implied by the sources even when the wording differs. It trades
      the determinism of the lexical checker for more permissive coverage and
      requires an on-disk model (downloaded once on first use).
"""

from __future__ import annotations

import re
from statistics import mean
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..models import RetrievedChunk

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
        "as", "at", "by", "from", "we", "you", "they", "i", "not", "but",
        "if", "so", "no", "yes", "do", "does", "did", "can", "could", "should",
        "will", "would", "may", "might", "have", "has", "had", "there", "which",
        "than", "about", "against", "between", "over", "into", "during",
    }
)


class GroundedTextCheck(Protocol):
    """Checks whether answer text is grounded in the retrieved evidence."""

    def check(
        self,
        answer: str,
        retrieved: list[RetrievedChunk],
        *,
        threshold: float,
    ) -> tuple[bool, dict[str, float]]:
        """Return ``(grounded, ratios)`` where ratios describe per-term coverage."""
        ...


class NaiveGroundedTextCheck:
    """Token-overlap grounding checker.

    For each substantive token in the answer we test whether it appears in any
    retrieved chunk. The grounded ratio = grounded tokens / total substantive
    tokens. A response is "grounded" when that ratio meets the threshold.
    """

    def __init__(self, *, min_meaningful_len: int = 3) -> None:
        self._min_len = min_meaningful_len

    def check(
        self,
        answer: str,
        retrieved: list[RetrievedChunk],
        *,
        threshold: float,
    ) -> tuple[bool, dict[str, float]]:
        if not answer.strip():
            return True, {"grounded_ratio": 1.0, "substantive_tokens": 0}

        corpus_text = " ".join(r.chunk.text for r in retrieved).lower()
        answer_terms = self._substantive_tokens(answer)

        if len(answer_terms) == 0:
            return True, {"grounded_ratio": 1.0, "substantive_tokens": 0}

        grounded = sum(1 for term in answer_terms if term in corpus_text)
        ratio = grounded / len(answer_terms)
        ratios: dict[str, float] = {
            "grounded_ratio": round(ratio, 4),
            "substantive_tokens": float(len(answer_terms)),
            "grounded_tokens": float(grounded),
        }
        return ratio >= threshold, ratios

    def _substantive_tokens(self, text: str) -> set[str]:
        tokens: set[str] = set()
        for match in re.finditer(r"\b[a-z0-9_.]+\b", text.lower()):
            token = match.group(0)
            # Skip stopwords and very short tokens, but always keep numbers.
            is_number = token.replace(".", "", 1).isdigit()
            if is_number or (len(token) >= self._min_len and token not in _STOPWORDS):
                tokens.add(token)
        return tokens


class EntailmentScorer(Protocol):
    """Scores premise-hypothesis pairs for semantic entailment.

    ``score_pairs`` takes a list of ``(premise, hypothesis)`` text pairs and,
    for each pair, returns a list of floating point scores — one per *label*
    (e.g. ``contradiction``, ``entailment``, ``neutral``). The position of each
    label in the returned list is model-specific; callers must map scores to
    labels via :attr:`labels` (or the model's ``id2label`` config).
    """

    @property
    def labels(self) -> list[str]:
        """Label names in score order (index ``i`` => label ``labels[i]``)."""
        ...

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[list[float]]:
        """Return, for each input pair, a list of raw scores per label."""
        ...


class CrossEncoderEntailmentScorer:
    """NLI cross-encoder scorer backed by ``sentence_transformers.CrossEncoder``.

    Uses a local *natural-language-inference* cross-encoder (default
    ``typeform/distilbert-base-uncased-mnli``, a public distilbert MNLI
    cross-encoder; gated alternatives exist) to score whether a hypothesis is
    entailed by a premise. The model weights are downloaded from the Hugging
    Face Hub the first time the scorer is actually used (~250MB, local download),
    not when the object is constructed.
    """

    def __init__(self, model_name: str = "typeform/distilbert-base-uncased-mnli") -> None:
        self.model_name = model_name
        # The model is loaded lazily on first use so that constructing a scorer
        # (and wiring it into tests/builders) never triggers a download or
        # spawns network traffic.
        self._model = None

    @property
    def labels(self) -> list[str]:
        """Label names in score order, derived from the model's ``id2label``."""
        self._ensure_loaded()
        assert self._model is not None
        id2label = self._model.model.config.id2label
        return [id2label[idx] for idx in sorted(id2label)]

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Imported lazily so importing this module never requires the heavy
        # sentence-transformers dependency to be resolvable/imported.
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[list[float]]:
        self._ensure_loaded()
        assert self._model is not None
        # predict() batches internally and truncates long sequences to the
        # model's max length, so we can rely on its built-in truncation.
        return self._model.predict(pairs).tolist()


class EntailmentGroundedTextCheck:
    """Semantic, entailment-based grounding checker (experimental).

    Unlike :class:`NaiveGroundedTextCheck`, which demands that the answer's
    *tokens* literally appear in the sources, this checker accepts an answer
    that is *logically entailed* by a retrieved chunk even when the phrasing
    differs. For each retrieved chunk it scores the pair
    ``(r.chunk.text, answer)`` with an NLI cross-encoder and takes the
    entailment label score: the premise is the retrieved chunk and the
    hypothesis is the answer, i.e. we ask whether the evidence entails the
    answer (``(premise, hypothesis)`` is the convention of
    the NLI cross-encoder; the default is the public distilbert MNLI model —
    gated alternatives exist).

    The answer is grounded when the *best* entailment score across sources
    meets the threshold. Being best-source (max) rather than average keeps a
    single strong source from being diluted by weak/irrelevant ones.

    Trade-offs vs. the lexical checker:

        * More permissive: accepts conclusions implied by the evidence even
          when the vocabulary does not overlap.
        * Non-deterministic-feeling: depends on a neural model.
        * Requires the model weights to be downloaded once (~250MB) on first
          use; kept local, never uploaded.

    The model is loaded lazily, so building an instance never downloads
    anything — only the first :meth:`check` call materialises the scorer.

    The entailment label is looked up case-insensitively because some
    checkpoints (e.g. ``typeform/distilbert-base-uncased-mnli``) expose labels
    in UPPERCASE (``ENTAILMENT``/``NEUTRAL``/``CONTRADICTION``) instead of
    lowercase.
    """

    def __init__(
        self,
        scorer: EntailmentScorer | None = None,
    ) -> None:
        self._scorer: EntailmentScorer = scorer or CrossEncoderEntailmentScorer()

    def check(
        self,
        answer: str,
        retrieved: list[RetrievedChunk],
        *,
        threshold: float,
    ) -> tuple[bool, dict[str, float]]:
        if not answer.strip():
            return True, {
                "entailment_max": 1.0,
                "entailment_mean": 1.0,
                "n_sources": float(len(retrieved)),
                "threshold": threshold,
            }

        if not retrieved:
            return False, {
                "entailment_max": 0.0,
                "entailment_mean": 0.0,
                "n_sources": 0.0,
                "threshold": threshold,
            }

        labels = self._scorer.labels
        # Case-insensitive lookup: some checkpoints (e.g.
        # typeform/distilbert-base-uncased-mnli) expose labels in UPPERCASE
        # (ENTAILMENT/NEUTRAL/CONTRADICTION) instead of lowercase.
        entailment_idx = next(
            (i for i, label in enumerate(labels) if label.strip().lower() == "entailment"),
            None,
        )
        if entailment_idx is None:
            raise ValueError(
                f"Entailment scorer does not expose an 'entailment' label; "
                f"got {labels}. Cannot compute semantic grounding."
            )

        pairs: list[tuple[str, str]] = [
            (r.chunk.text, answer) for r in retrieved
        ]
        # Each pair is (premise, hypothesis): premise = retrieved chunk,
        # hypothesis = answer — we ask whether the evidence entails the answer.
        scores = self._scorer.score_pairs(pairs)

        ent_scores: list[float] = [
            float(pair_scores[entailment_idx]) for pair_scores in scores
        ]
        max_ent = max(ent_scores, default=0.0)
        mean_ent = mean(ent_scores) if ent_scores else 0.0

        result: dict[str, float] = {
            "entailment_max": round(max_ent, 4),
            "entailment_mean": round(mean_ent, 4),
            "n_sources": float(len(retrieved)),
            "threshold": threshold,
        }
        return max_ent >= threshold, result
