"""WebAnno ↔ spaCy Converter.

Convert between WebAnno TSV format and spaCy training data formats.
Supports NER, NEL, POS tagging, and lemmatization.
"""

__version__ = "0.2.0"
__author__ = "SasaP"

# Public API - convenient imports
from .parsers.tsv_parser_v3 import WebAnnoNELParser, WebAnnoLEXISParser
from .converters.webanno_to_spacy import AnnotationSentencesToDocBinConverterV2
from .converters.spacy_to_webanno import DocBinToAnnotationSentencesConverter
from .writers.webanno_writer import WebAnnoNELWriter
from .models.annotation_sentence import AnnotationSentence
from .models.annotation_token import AnnotationToken

__all__ = [
    "WebAnnoNELParser",
    "WebAnnoLEXISParser",
    "AnnotationSentencesToDocBinConverterV2",
    "DocBinToAnnotationSentencesConverter",
    "WebAnnoNELWriter",
    "AnnotationSentence",
    "AnnotationToken",
    "__version__",
]
