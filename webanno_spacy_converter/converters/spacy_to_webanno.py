from typing import List, Dict, Tuple
from spacy.tokens import DocBin, Doc
from webanno_spacy_converter.models.annotation_sentence import AnnotationSentence
from webanno_spacy_converter.models.annotation_token import AnnotationToken
from collections import defaultdict

class DocBinToAnnotationSentencesConverter:
    """
    Converts spaCy DocBin or Doc objects into a list of AnnotationSentence objects
    suitable for WebAnno export.
    """

    def __init__(self, nlp):
        self.nlp = nlp

    def convert(self, docbin_path: str) -> List[AnnotationSentence]:
        """Convert from a DocBin file on disk."""
        doc_bin = DocBin().from_disk(docbin_path)
        docs = list(doc_bin.get_docs(self.nlp.vocab))
        return self.convert_docs(docs)

    def convert_docbin(self, docbin: DocBin) -> List[AnnotationSentence]:
        """Convert directly from a DocBin object."""
        docs = list(docbin.get_docs(self.nlp.vocab))
        return self.convert_docs(docs)

    def convert_docs(self, docs: List[Doc]) -> List[AnnotationSentence]:
        """Convert from a list of Doc objects."""
        sentences: List[AnnotationSentence] = []
        sentence_index = 1
        group_counter = 1

        for doc in docs:
            # Map entity spans to tokens
            entity_map: Dict[Tuple[int, int], Tuple[str, str]] = {}
            group_ids: Dict[Tuple[int, int], str] = {}

            for ent in doc.ents:
                span = (ent.start_char, ent.end_char)
                label = ent.label_
                kb_id = getattr(ent, 'kb_id_', getattr(ent, 'ent_kb_id_', None))
                if kb_id is None or kb_id == "NIL":
                    kb_id = "*"
                if ent.end - ent.start > 1:
                    group = str(group_counter)
                    group_ids[span] = group
                    group_counter += 1
                entity_map[span] = (label, kb_id)

            for sent in doc.sents:
                tokens: List[AnnotationToken] = []
                min_start = sent.start_char
                for i, token in enumerate(sent):
                    abs_start = token.idx
                    abs_end = token.idx + len(token.text)
                    rel_start = abs_start - min_start
                    rel_end = abs_end - min_start

                    layer_data = {}
                    for (span_start, span_end), (label, kb_id) in entity_map.items():
                        if span_start <= abs_start < span_end:
                            if (span_start, span_end) in group_ids:
                                g = group_ids[(span_start, span_end)]
                                layer_data['value'] = f"{label}[{g}]"
                                layer_data['identifier'] = f"{kb_id}[{g}]" 
                            else:
                                layer_data['value'] = label
                                layer_data['identifier'] = kb_id
                            break

                    tok = AnnotationToken(
                        sentence_index=sentence_index,
                        token_index=i + 1,
                        text=token.text,
                        start=rel_start,
                        end=rel_end,
                        layers=layer_data
                    )
                    tokens.append(tok)

                sent_obj = AnnotationSentence(
                    text=sent.text,
                    tokens=tokens
                )
                sentences.append(sent_obj)
                sentence_index += 1

        return sentences
