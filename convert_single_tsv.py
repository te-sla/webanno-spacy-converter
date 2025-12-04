from pathlib import Path

import spacy
from spacy.tokens import DocBin

from webanno_spacy_converter.parsers.tsv_parser_v3 import WebAnnoNELParser
from webanno_spacy_converter.converters.webanno_to_spacy import (
    AnnotationSentencesToDocBinConverterV2,
)


def main() -> None:
    base = Path(__file__).parent
    tsv_path = base / "test_data" / "sr-elexis-WSD_0001_0500.tsv"
    out_path = base / "test_data" / "sr_elexis_debug.spacy"

    print(f"Reading TSV: {tsv_path}")
    parser = WebAnnoNELParser(str(tsv_path))
    sentences = parser.parse()
    print(f"Parsed {len(sentences)} sentences")

    print("Loading spaCy pipeline 'my_nlp_el_cnn1'...")
    nlp = spacy.load(base / "my_nlp_el_cnn1")

    # Use correct WebAnno layer names for this corpus
    converter = AnnotationSentencesToDocBinConverterV2(
        nlp,
        sentences_per_doc=10,
        tag_layer="coarseValue",  # POS
        lemma_layer="value_4",    # lemma
        ner=True,
        nel=True,
    )

    docbin: DocBin = converter.convert(sentences)
    docbin.to_disk(str(out_path))
    print(f"Wrote DocBin to: {out_path}")


if __name__ == "__main__":
    main()
