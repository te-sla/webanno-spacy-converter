from typing import List
from abc import ABC, abstractmethod
from webanno_spacy_converter.models.annotation_sentence import AnnotationSentence
from webanno_spacy_converter.models.annotation_token import AnnotationToken

class BaseWebAnnoTSVWriter(ABC):
    """
    Abstract base class for writing annotations to the WebAnno TSV 3.x format.

    This class provides a framework for writing a list of annotated sentences
    into a TSV file compatible with WebAnno. It handles sentence structure,
    token positioning, and delegates the formatting of annotation layers to subclasses.

    Attributes:
        sentences (List[AnnotationSentence]): List of annotated sentence objects to write.
    """

    def __init__(self, sentences: List[AnnotationSentence]):
        self.sentences = sentences

    def save(self, output_path: str) -> None:
        """
        Save the annotations to a TSV file at the specified path.

        The format includes the WebAnno format header, the annotation layer specification,
        and one block per sentence with token annotations.

        Args:
            output_path (str): Path to the file where output should be written.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # WebAnno requires this header for file format version
            f.write("#FORMAT=WebAnno TSV 3.3\n")
            # Layer header defined by subclass
            f.write(self._build_layer_header() + "\n\n")

            offset = 0  # cumulative character offset
            for i, sentence in enumerate(self.sentences, start=1):
                f.write("\n")
                f.write(f"#Text={sentence.text}\n")
                for token in sentence.tokens:
                    layers = self._format_token_layers(token)
                    abs_start = token.start + offset
                    abs_end = token.end + offset
                    line = f"{token.sentence_index}-{token.token_index}\t{abs_start}-{abs_end}\t{token.text}\t" + "\t".join(layers) + "\t"
                    f.write(line + "\n")
                offset += len(sentence.text) + 1

    @abstractmethod
    def _build_layer_header(self) -> str:
        """
        Construct the layer header string used by WebAnno.

        Returns:
            str: Header line describing the annotation layers.
        """
        pass

    @abstractmethod
    def _format_token_layers(self, token: AnnotationToken) -> List[str]:
        """
        Format the annotation layers for a single token.

        Args:
            token (AnnotationToken): Token with annotation data.

        Returns:
            List[str]: List of string fields representing layers.
        """
        pass

class WebAnnoNELWriter(BaseWebAnnoTSVWriter):
    """
    WebAnno TSV writer for NEL tasks.

    Supports writing two annotation layers per token:
        - 'identifier': the Wikidata entity URL or ID
        - 'value': the NER label (e.g., LOC, PER)

    The output format aligns with the structure required for
    Named Entity Linking annotation projects.
    """

    def _build_layer_header(self) -> str:
        """
        Build the layer header for NEL output.

        Returns:
            str: TSV layer definition line.
        """
        return "#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value"

    def _format_token_layers(self, token: AnnotationToken) -> List[str]:
        """
        Extract NEL-specific annotation data for a token.

        Args:
            token (AnnotationToken): Token object with layers.

        Returns:
            List[str]: A list with [identifier, value] fields.
        """
        identifier = token.layers.get("identifier", "_")
        value = token.layers.get("value", "_")

        if not identifier.startswith("*") and not identifier.startswith("_") and len(identifier)>0 and not identifier.startswith("http"):
            identifier = f"http://www.wikidata.org/entity/{identifier}"
        
        if len(identifier) == 0:
            if value == "_":
                identifier = "_"
            else:
                identifier = "*"

        return [identifier, value]
