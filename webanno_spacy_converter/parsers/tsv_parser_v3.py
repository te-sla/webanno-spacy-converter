from abc import ABC
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from typing import DefaultDict
from ..models.annotation_token import AnnotationToken
from ..models.annotation_sentence import AnnotationSentence
from webanno_spacy_converter.models.sentence_with_mwes import MultiWordExpression, AnnotatedSentenceWithMWEs


class BaseWebAnnoTSVParser(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.header_lines: List[str] = []
        self.layer_names: Dict[int, str] = {}
        self.sentences: List[AnnotationSentence] = []

    def load_lines(self) -> List[str]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def parse(self) -> List[AnnotationSentence]:
        lines = self.load_lines()
        self._extract_headers(lines)
        sentence_blocks = self._split_sentences(lines)
        # No need to pass sentence index, as it is read from each token line
        self.sentences = [self._parse_sentence_lines(block) for block in sentence_blocks]
        return self.sentences

    def _extract_headers(self, lines: List[str]) -> None:
        col_index = 0
        for line in lines:
            if line.startswith("#T_SP="):
                cleaned = line[len("#T_SP="):]
                parts = cleaned.split('|')
                for name in parts[1:]:  # skip the type name
                    name = name.strip()
                    if name in self.layer_names.values():
                        name = f"{name}_{col_index}"
                    self.layer_names[col_index] = name.strip()
                    col_index += 1

    def _split_sentences(self, lines: List[str]) -> List[List[str]]:
        blocks = []
        current_block = []
        for line in lines:
            if line.startswith("#Text="):
                if current_block:
                    blocks.append(current_block)
                current_block = [line]
            elif not line.startswith("#"):
                current_block.append(line)
        if current_block:
            blocks.append(current_block)
        return blocks

    def _parse_sentence_lines(self, sentence_lines: List[str]) -> AnnotationSentence:
        sentence_text = sentence_lines[0][6:]  # Remove "#Text="
        token_lines = sentence_lines[1:]
        min_start_index = None
        tokens = []

        for token_index, line in enumerate(token_lines, start=1):
            token, min_start_index = self._parse_token_line(
                line, token_index, min_start_index
            )
            tokens.append(token)

        return self._finalize_sentence(sentence_text, tokens)

    def _parse_token_line(
        self,
        line: str,
        token_index: int,
        min_start_index: Optional[int],
    ) -> Tuple[AnnotationToken, int]:
        parts = line.split('\t')
        if len(parts) < 3:
            raise ValueError(f"Malformed token line: {line}")
        # Extract sentence index from the first column (e.g., '1-1')
        sent_token = parts[0]
        try:
            sentence_index = int(sent_token.split('-')[0])
        except Exception as e:
            raise ValueError(f"Invalid sentence index in token: {sent_token}") from e
        position = parts[1]
        token_text = parts[2]

        if '-' in position:
            start, end = map(int, position.split("-"))
        else:
            start = end = int(position)

        if min_start_index is None:
            min_start_index = start

        offset_start = start - min_start_index
        offset_end = end - min_start_index

        layers = {}
        for i, col in enumerate(parts[3:]):
            if col != "_":
                layer_name = self.layer_names.get(i, f"layer{i}")
                layers[layer_name] = col
        
        token = AnnotationToken(
            sentence_index=sentence_index,
            token_index=token_index,
            text=token_text,
            start=offset_start,
            end=offset_end,
            layers=layers
        )

        return token, min_start_index

    def _finalize_sentence(self, sentence_text: str, tokens: List[AnnotationToken]) -> AnnotationSentence:
        return AnnotationSentence(
            text=sentence_text,
            tokens=tokens,
            entities=[],
        )

class WebAnnoNELParser(BaseWebAnnoTSVParser):
    def _finalize_sentence(self, sentence_text: str, tokens: List[AnnotationToken]) -> AnnotationSentence:
        entities: List[Tuple[int, int, str, str]] = []
        grouped: DefaultDict[str, List[Tuple[int, int, str, str]]] = defaultdict(list)

        for token in tokens:
            ner_layer = token.layers.get("value")
            nel_layer = token.layers.get("identifier")

            if ner_layer and nel_layer:
                # Multi-token grouped entity
                if "[" in ner_layer and "[" in nel_layer:
                    label, group_id_ner = ner_layer.split("[")
                    qid_url, group_id_link = nel_layer.split("[")
                    group_id_ner = group_id_ner.rstrip("]")
                    group_id_link = group_id_link.rstrip("]")
                    qid = qid_url.rsplit("/", 1)[-1]

                    if group_id_ner == group_id_link:
                        grouped[group_id_ner].append((token.start, token.end, label, qid))

                # Single-token entity
                elif "[" not in ner_layer and "[" not in nel_layer:
                    label = ner_layer
                    qid = nel_layer.rsplit("/", 1)[-1]
                    entities.append((token.start, token.end, label, qid))

        for group in grouped.values():
            starts = [s for s, _, _, _ in group]
            ends = [e for _, e, _, _ in group]
            label = group[0][2]
            qid = group[0][3]
            entities.append((min(starts), max(ends), label, qid))

        return AnnotationSentence(
            text=sentence_text,
            tokens=tokens,
            entities=entities,
        )
class WebAnnoLEXISParser(WebAnnoNELParser):
    """
    Extends WebAnnoNELParser to include multi-word expression (MWE) extraction from the LEXIS corpus.
    """

    def _finalize_sentence(self, sentence_text: str, tokens: List[AnnotationToken]) -> AnnotatedSentenceWithMWEs:
        base = super()._finalize_sentence(sentence_text, tokens)

        mwe_groups: DefaultDict[str, List[Tuple[int, AnnotationToken]]] = defaultdict(list)
        mwe_lemmas: DefaultDict[str, str] = defaultdict(str)
        mwe_types: DefaultDict[str, str] = defaultdict(str)

        for idx, token in enumerate(tokens):
            mwe_id = token.layers.get("MWEid")
            mwe_lemma = token.layers.get("MWElemma")
            mwe_type = token.layers.get("MWEtype")

            if mwe_id and mwe_id != "_":
                group_id = mwe_id.split("[")[0] if "[" in mwe_id else mwe_id
                mwe_groups[group_id].append((idx, token))

                # Prefer non-* lemma if available
                if mwe_lemma != "*":
                    mwe_lemmas[group_id] = mwe_lemma.split("[")[0].strip()


                if mwe_type != "*":
                    mwe_types[group_id] = mwe_type.split("[")[0].strip()


        mwes: List[MultiWordExpression] = []
        for group_id, token_list in mwe_groups.items():
            token_indices = [idx for idx, _ in token_list]
            lemma = mwe_lemmas.get(group_id, "*")
            mwe_type = mwe_types.get(group_id, "")
            mwes.append(MultiWordExpression(
                lemma=lemma,
                token_count=len(token_indices),
                token_indices=token_indices,
                type=mwe_type,
                group_id=group_id
            ))

        return AnnotatedSentenceWithMWEs(
            text=base.text,
            tokens=base.tokens,
            entities=base.entities,
            mwes=mwes
        )
