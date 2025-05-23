from ..parsers.tsv_parser_v3 import WebAnnoNELParser
from ..writers.webanno_writer import WebAnnoNELWriter

def pretty_print_sentences(sentences):
    for sentence in sentences:
        print(f"Sentence: {sentence.text}")
        for start, end, entity_type, link in sentence.entities:
            print(f"  Entity: {sentence.text[start:end]} (Type: {entity_type}, Link: {link})")

def main():
    parser = WebAnnoNELParser("alzir.tsv")
    parser.parse()
    pretty_print_sentences(parser.sentences)
    writer = WebAnnoNELWriter(parser.sentences)
    writer.save("output.tsv")

if __name__ == "__main__":
    main()

