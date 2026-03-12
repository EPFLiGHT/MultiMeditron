"""
NLLB-200 Translator with Language Detection

This module provides a translation interface using the NLLB-200 finetuned multilingual model
combined with fastText language detection.

Translation Strategy:
- High confidence detection (>80%): Translate question to English, process, then translate back
- Low confidence detection (≤80%): Pass through as-is to avoid mistranslation of ambiguous text

The confidence threshold helps prevent incorrect translations of short or ambiguous text
while enabling accurate translation of clearly detected languages.

Usage:
    translator = NLLBTranslator()
    english_text, user_lang = translator.translate_to_english(
        user_question, return_detected_lang=True
    )
    response_in_user_lang = translator.translate_from_english(
        english_response, tgt_lang=user_lang
    )
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download

LOGGER = logging.getLogger(__name__)
DEFAULT_BASE_MODEL = "facebook/nllb-200-3.3B"
DEFAULT_LOCAL_MODEL = (
    Path(__file__).resolve().parent / "models" / "nllb-consensus-finetuned-1epoch"
)


class NLLBTranslator:
    """NLLB-200 translator with fastText language detection."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        lang_detect_model: str = "facebook/fasttext-language-identification",
    ):
        """Initialize NLLB translator with fastText language detection."""
        if model_name is None:
            if DEFAULT_LOCAL_MODEL.exists():
                model_name = str(DEFAULT_LOCAL_MODEL)
                LOGGER.info("Using local fine-tuned NLLB model at %s", model_name)
            else:
                model_name = DEFAULT_BASE_MODEL
                LOGGER.info(
                    "Local fine-tuned model not found. Falling back to %s",
                    DEFAULT_BASE_MODEL,
                )

        LOGGER.info("Loading NLLB model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        LOGGER.info("Loading fastText language detection model from %s", lang_detect_model)
        try:
            import fasttext

            model_path = hf_hub_download(
                repo_id=lang_detect_model,
                filename="model.bin"
            )
            fasttext.FastText.eprint = lambda x: None
            self.lang_detector = fasttext.load_model(model_path)
            LOGGER.info("fastText model loaded successfully")
        except Exception as e:
            LOGGER.exception("Failed to load fastText language detector: %s", e)
            LOGGER.info("Ensure dependencies are installed: `pip install \"numpy<2.0\" fasttext`")
            raise

        LOGGER.info("NLLB translator ready on %s", self.device)

    def detect_language(self, text: str, confidence_threshold=0.80) -> str:
        """
        Detect language using fastText. Returns 'eng_Latn' if confidence < threshold
        to trigger pass-through behavior (no translation).
        """
        try:
            clean_text = text.replace('\n', ' ').strip()
            predictions = self.lang_detector.predict(clean_text, k=3)

            detected_code = predictions[0][0].replace('__label__', '')
            confidence = float(predictions[1][0])

            LOGGER.debug("Detected language %s (confidence %.3f)", detected_code, confidence)

            if confidence < confidence_threshold:
                LOGGER.warning(
                    "Low confidence language detection (%.3f < %.3f). Falling back to eng_Latn.",
                    confidence,
                    confidence_threshold,
                )
                for i in range(min(3, len(predictions[0]))):
                    alt_code = predictions[0][i].replace('__label__', '')
                    alt_conf = float(predictions[1][i])
                    LOGGER.warning("Alternative prediction %d: %s (%.3f)", i + 1, alt_code, alt_conf)
                return 'eng_Latn'

            try:
                token_id = self.tokenizer.convert_tokens_to_ids(detected_code)
                if token_id == self.tokenizer.unk_token_id:
                    LOGGER.warning(
                        "Detected language code '%s' is not supported by tokenizer. Falling back to eng_Latn.",
                        detected_code,
                    )
                    return 'eng_Latn'
            except Exception:
                LOGGER.warning(
                    "Tokenizer validation failed for detected language code '%s'. Falling back to eng_Latn.",
                    detected_code,
                )
                return 'eng_Latn'

            return detected_code

        except Exception as e:
            LOGGER.exception("Language detection failed (%s). Falling back to eng_Latn.", e)
            return 'eng_Latn'

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text from src_lang to tgt_lang using NLLB."""
        if not text or not text.strip():
            return text

        if src_lang == tgt_lang:
            return text

        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        try:
            forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        except Exception as e:
            LOGGER.error("Failed to get token ID for %s: %s", tgt_lang, e)
            return text

        with torch.no_grad():
            translated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
        
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        result = decoded[0]

        text_preview = text[:80] + '...' if len(text) > 80 else text
        result_preview = result[:80] + '...' if len(result) > 80 else result
        LOGGER.debug("Translation input preview: %s", text_preview)
        LOGGER.debug("Translation output preview: %s", result_preview)

        return result

    def translate_to_english(
        self,
        text: str,
        src_lang: Optional[str] = None,
        return_detected_lang: bool = False,
    ) -> Union[str, Tuple[str, str]]:
        """
        Translate to English if high confidence detection, otherwise pass through.

        Stateless usage:
            translated, detected_lang = translate_to_english(text, return_detected_lang=True)
        """
        if src_lang is None:
            src_lang = self.detect_language(text)

        if src_lang == 'eng_Latn':
            translated = text
        else:
            translated = self.translate(text, src_lang, 'eng_Latn')

        if return_detected_lang:
            return translated, src_lang
        return translated

    def translate_from_english(self, text: str, tgt_lang: Optional[str] = None) -> str:
        """
        Translate from English to a caller-provided target language.
        """
        if tgt_lang is None:
            raise ValueError(
                "tgt_lang is required for stateless translation. "
                "Use `translate_to_english(..., return_detected_lang=True)` and pass "
                "the detected language into `translate_from_english`."
            )

        if tgt_lang == 'eng_Latn':
            return text

        return self.translate(text, 'eng_Latn', tgt_lang)
