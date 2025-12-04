from pathlib import Path

from webanno_spacy_converter.parsers.tsv_parser_v3 import WebAnnoNELParser


def main() -> None:
    tsv_path = Path("test_data") / "sr-elexis-WSD_0001_0500.tsv"
    print(f"Reading: {tsv_path}")
    parser = WebAnnoNELParser(str(tsv_path))
    sentences = parser.parse()
    print(f"Parsed {len(sentences)} sentences")
    if not sentences:
        return

    first = sentences[0]
    print("First sentence:", first.text)
    if not first.tokens:
        print("No tokens in first sentence")
        return

    print("Layers for each token in first sentence (name -> value):")
    for tok in first.tokens:
        print(f"Token '{tok.text}': {tok.layers}")


if __name__ == "__main__":
    main()
