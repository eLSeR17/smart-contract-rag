"""Tests for the anti-hallucination grounding checker."""

from __future__ import annotations

from smart_contract_rag.builders import build_grounding_check
from smart_contract_rag.config import Settings
from smart_contract_rag.generation.grounding import (
    CrossEncoderEntailmentScorer,
    EntailmentGroundedTextCheck,
    NaiveGroundedTextCheck,
)
from smart_contract_rag.models import Chunk, RetrievedChunk, SourceRef


def _retrieved(*texts: str) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(text=t, source=SourceRef(doc_id="doc1")),
            score=0.9,
        )
        for t in texts
    ]


CORPUS = _retrieved(
    "The withdraw function is vulnerable to reentrancy and must use checks-effects-interactions.",
    "Aave v3 uses a liquidity index of 1.05.",
)


class TestNaiveGroundedTextCheck:
    def test_grounded_response_accepted(self) -> None:
        checker = NaiveGroundedTextCheck()
        ok, ratios = checker.check(
            "The withdraw function is vulnerable to reentrancy.",
            CORPUS,
            threshold=0.5,
        )
        assert ok is True
        assert ratios["grounded_ratio"] >= 0.5

    def test_hallucinated_response_rejected(self) -> None:
        checker = NaiveGroundedTextCheck()
        # Answer talks about an unrelated crypto exchange that is not in corpus.
        ok, ratios = checker.check(
            "Binance listed a new token called XYZ next week.",
            CORPUS,
            threshold=0.5,
        )
        assert ok is False
        assert ratios["grounded_ratio"] < 0.5

    def test_empty_answer_is_grounded(self) -> None:
        checker = NaiveGroundedTextCheck()
        ok, _ = checker.check("", CORPUS, threshold=0.5)
        assert ok is True

    def test_number_grounding(self) -> None:
        checker = NaiveGroundedTextCheck()
        # The number 1.05 is grounded; a fabricated number is not.
        ok, ratios = checker.check("The liquidity index is 1.05.", CORPUS, threshold=0.5)
        assert ok is True
        assert ratios["grounded_tokens"] >= 1

        ok_false, _ = checker.check("The fee rate is 42.7 percent now.", CORPUS, threshold=0.8)
        # 42.7 is not in the corpus, pushing ratio down.
        assert ok_false is False

    def test_threshold_controls_strictness(self) -> None:
        checker = NaiveGroundedTextCheck()
        answer = "The withdraw function is vulnerable to reentrancy and exploits are likely."
        # Full grounding yields high ratio; a strict threshold passes.
        ok_strict, ratios = checker.check(answer, CORPUS, threshold=0.5)
        assert ok_strict is True
        # An impossible threshold should reject even a grounded answer.
        ok_impossible, _ = checker.check(answer, CORPUS, threshold=0.99)
        assert ok_impossible is False

    def test_no_retrieved_evidence_flagged(self) -> None:
        checker = NaiveGroundedTextCheck()
        ok, ratios = checker.check("Anything at all here.", [], threshold=0.5)
        assert ok is False
        assert ratios["grounded_ratio"] == 0.0


class FakeEntailmentScorer:
    """Scripted entailment scorer with a label order.

    ``scripted`` is a list of label-score lists, one per pair. If a call has
    more pairs than scripted, the script cycles. Exposes ``labels`` in score
    order exactly like the real cross-encoder.
    """

    def __init__(
        self,
        labels: list[str],
        scripted: list[list[float]],
    ) -> None:
        self._labels = list(labels)
        self._scripted = scripted

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[list[float]]:
        return [
            list(self._scripted[i % len(self._scripted)])
            for i in range(len(pairs))
        ]


_DEFAULT_LABELS = ["contradiction", "entailment", "neutral"]


def _entailment_checker(
    *scripted: list[float],
    labels: list[str] | None = None,
) -> EntailmentGroundedTextCheck:
    scorer = FakeEntailmentScorer(labels or _DEFAULT_LABELS, list(scripted))
    return EntailmentGroundedTextCheck(scorer)


class TestEntailmentGroundedTextCheck:
    def test_entailing_answer_accepted(self) -> None:
        checker = _entailment_checker(
            [0.1, 0.2, 0.7],  # source 1: neutral dominates
            [0.1, 0.9, 0.0],  # source 2: entails
        )
        ok, ratios = checker.check(
            "The withdraw function must follow checks-effects-interactions.",
            CORPUS,
            threshold=0.5,
        )
        assert ok is True
        assert ratios["entailment_max"] == 0.9
        assert ratios["entailment_mean"] == 0.55
        assert ratios["n_sources"] == 2.0

    def test_answer_rejected_when_neutral_dominates(self) -> None:
        checker = _entailment_checker(
            [0.1, 0.2, 0.7],
            [0.1, 0.2, 0.7],
        )
        ok, ratios = checker.check(
            "The token will 100x next week.",
            CORPUS,
            threshold=0.5,
        )
        assert ok is False
        assert ratios["entailment_max"] == 0.2
        assert ratios["entailment_mean"] == 0.2

    def test_empty_answer_accepted(self) -> None:
        checker = _entailment_checker([0.1, 0.2, 0.7])
        ok, ratios = checker.check("", CORPUS, threshold=0.5)
        assert ok is True
        assert ratios["entailment_max"] == 1.0
        assert ratios["entailment_mean"] == 1.0
        assert ratios["n_sources"] == 2.0

    def test_no_evidence_rejected_with_zero_max(self) -> None:
        checker = _entailment_checker([0.1, 0.2, 0.7])
        ok, ratios = checker.check("Anything at all.", [], threshold=0.5)
        assert ok is False
        assert ratios["entailment_max"] == 0.0
        assert ratios["entailment_mean"] == 0.0
        assert ratios["n_sources"] == 0.0

    def test_best_source_semantics_not_average(self) -> None:
        # Mean entailment (0.3 + 0.9) / 2 = 0.6 < 0.7, but the best source
        # entails with 0.9 -> accepted. A mean-based checker would refuse.
        checker = _entailment_checker(
            [0.01, 0.3, 0.69],
            [0.01, 0.9, 0.09],
        )
        ok, ratios = checker.check(
            "The liquidity index handling is correct.",
            CORPUS,
            threshold=0.7,
        )
        assert ok is True
        assert ratios["entailment_max"] == 0.9
        assert ratios["entailment_mean"] == 0.6

    def test_threshold_boundary(self) -> None:
        # Entailment score is exactly 0.5: accepted at threshold 0.5, rejected
        # at 0.51.
        checker = _entailment_checker([0.0, 0.5, 0.5])
        ok_boundary, _ = checker.check("Boundary answer.", CORPUS, threshold=0.5)
        assert ok_boundary is True
        ok_stricter, _ = checker.check("Boundary answer.", CORPUS, threshold=0.51)
        assert ok_stricter is False

    def test_lazy_loading_no_model_on_construction(self) -> None:
        scorer = CrossEncoderEntailmentScorer()
        assert scorer._model is None

    def test_labels_resolved_by_name_not_index(self) -> None:
        # The default NLI id2label order is (contradiction, entailment,
        # neutral), but the checker must not assume that: use a scorer with a
        # shuffled label order where entailment lives at index 2 and neutral at
        # index 1 with a *different* value, so reading the wrong column would
        # change the result.
        checker = _entailment_checker(
            [0.85, 0.05, 0.10],  # contradiction, neutral, entailment
            labels=["contradiction", "neutral", "entailment"],
        )
        scorer = checker._scorer
        assert isinstance(scorer, FakeEntailmentScorer)
        assert scorer.labels == ["contradiction", "neutral", "entailment"]
        ok, ratios = checker.check(
            "The withdraw function must follow checks-effects-interactions.",
            CORPUS,
            threshold=0.08,
        )
        assert ok is True
        # Entailment (idx 2) is 0.10; reading the neutral column (idx 1, 0.05)
        # by mistake would reject at threshold 0.08.
        assert ratios["entailment_max"] == 0.1

    def test_uppercase_labels_case_insensitive_lookup(self) -> None:
        # Regression: typeform/distilbert-base-uncased-mnli exposes labels as
        # ['ENTAILMENT', 'NEUTRAL', 'CONTRADICTION'] (UPPERCASE); the old exact
        # lookup labels.index("entailment") raised ValueError in live evals.
        # Scores follow the label order: entailment (idx 0) = 0.9.
        checker = _entailment_checker(
            [0.9, 0.05, 0.05],  # ENTAILMENT, NEUTRAL, CONTRADICTION
            labels=["ENTAILMENT", "NEUTRAL", "CONTRADICTION"],
        )
        ok, ratios = checker.check(
            "The withdraw function must follow checks-effects-interactions.",
            CORPUS,
            threshold=0.5,
        )
        assert ok is True
        # Entailment lives at idx 0 (0.9); no ValueError, score read correctly.
        assert ratios["entailment_max"] == 0.9
        assert ratios["entailment_mean"] == 0.9


class TestGroundingSettingsWiring:
    def test_semantic_mode_settings_parsed(self) -> None:
        settings = Settings.from_env(
            environ={
                "GROUNDING_MODE": "semantic",
                "GROUNDING_MODEL": "cross-encoder/nli-roberta-base",
                "GROUNDING_THRESHOLD": "0.7",
            }
        )
        assert settings.grounding_mode == "semantic"
        assert settings.grounding_model == "cross-encoder/nli-roberta-base"
        assert settings.grounding_threshold == 0.7
        assert isinstance(settings.grounding_threshold, float)

    def test_lexical_defaults(self) -> None:
        settings = Settings.from_env(environ={})
        assert settings.grounding_mode == "lexical"
        assert settings.grounding_model == "typeform/distilbert-base-uncased-mnli"
        assert settings.grounding_threshold == 0.5


class TestGroundingBuildersWiring:
    def test_build_semantic_check(self) -> None:
        settings = Settings.from_env(
            environ={"GROUNDING_MODE": "semantic", "GROUNDING_MODEL": "my-model"}
        )
        check = build_grounding_check(settings)
        assert isinstance(check, EntailmentGroundedTextCheck)
        assert isinstance(check._scorer, CrossEncoderEntailmentScorer)
        assert check._scorer.model_name == "my-model"
        # Constructing must not load/download anything.
        assert check._scorer._model is None

    def test_build_lexical_check(self) -> None:
        settings = Settings.from_env(environ={"GROUNDING_MODE": "lexical"})
        assert isinstance(build_grounding_check(settings), NaiveGroundedTextCheck)

    def test_build_invalid_mode_raises(self) -> None:
        settings = Settings.from_env(environ={"GROUNDING_MODE": "fuzzy"})
        try:
            build_grounding_check(settings)
        except ValueError as exc:
            assert "fuzzy" in str(exc)
            assert "lexical" in str(exc)
            assert "semantic" in str(exc)
        else:  # pragma: no cover - safety net if the raise disappears
            raise AssertionError("build_grounding_check should raise ValueError")
