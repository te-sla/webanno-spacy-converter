# WebAnno ↔ spaCy Converter

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![spaCy 3.x](https://img.shields.io/badge/spaCy-3.x-09a3d5.svg)](https://spacy.io)

A Python package for converting between [WebAnno](https://webanno.github.io/webanno/) TSV 3.3 format and [spaCy](https://spacy.io/) training data. Supports NER, NEL (Named Entity Linking), POS tagging, lemmatization, and Cyrillic↔Latin transliteration.

Perfect for researchers and NLP practitioners working on corpus conversion and annotation pipelines.

---

## 📑 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [GUI Usage](#️-gui-graphical-interface)
- [CLI Usage](#️-cli-command-line-interface)
- [Python API](#-python-api)
- [WebAnno TSV Format](#-webanno-tsv-format)
- [Configuration Options](#️-configuration-options)
- [Transliteration](#-transliteration)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **WebAnno → spaCy** | Convert annotated TSV files to spaCy's binary `.spacy` format |
| **spaCy → WebAnno** | Export spaCy Docs back to WebAnno TSV for re-annotation |
| **Named Entity Recognition** | Full NER support with entity types and spans |
| **Named Entity Linking** | Preserve Wikidata QIDs and other entity links |
| **POS Tagging & Lemmas** | Extract and convert morphological annotations |
| **Transliteration** | Cyrillic↔Latin conversion (Serbian, Montenegrin, Macedonian, Russian, etc.) |
| **Multi-Word Expressions** | LEXIS format support for MWE annotations |
| **Train/Dev/Test Splits** | Automatic 80/10/10 dataset splitting |
| **Shuffle Modes** | Chunk-level, sentence-level, or no shuffling |
| **Hidden Character Cleaning** | Remove zero-width characters that break transformers |
| **GUI & CLI** | Both graphical and command-line interfaces |

---

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install webanno-spacy-converter
```

### From Source

```bash
git clone https://github.com/te-sla/webanno-spacy-converter.git
cd webanno-spacy-converter
pip install -e .
```

### Dependencies

The package automatically installs:
- `spacy>=3.5` - NLP framework
- `cyrtranslit>=1.0` - Cyrillic transliteration

For development:
```bash
pip install webanno-spacy-converter[dev]
```

---

## 🚀 Quick Start

### 1. Convert WebAnno TSV to spaCy (Python)

```python
from webanno_spacy_converter import (
    WebAnnoNELParser, 
    AnnotationSentencesToDocBinConverterV2
)
import spacy

# Parse WebAnno TSV file
parser = WebAnnoNELParser("annotations.tsv")
parser.parse()
sentences = parser.sentences

# Convert to spaCy DocBin
nlp = spacy.blank("sr")  # or any language
converter = AnnotationSentencesToDocBinConverterV2(nlp, sentences_per_doc=3)
docbin = converter.convert(sentences)

# Save to disk
docbin.to_disk("output.spacy")
```

### 2. Launch the GUI

```bash
webanno-to-spacy-gui
```

Or from Python:
```python
from webanno_spacy_converter.gui import run_gui
run_gui()
```

### 3. Use the CLI

```bash
webanno-to-spacy ./input_folder ./output_folder
```

---

## 🖥️ GUI (Graphical Interface)

Launch the GUI with:
```bash
webanno-to-spacy-gui
```

### GUI Features

| Setting | Description |
|---------|-------------|
| **Input Mode** | Choose between folder or individual files |
| **Output Folder** | Where to save `.spacy` files |
| **Output Root** | Base name for output files (e.g., `dataset` → `dataset-train.spacy`) |
| **Model** | spaCy model (`blank:sr`, `blank:en`, or trained model name) |
| **Sentences per Doc** | How many sentences to combine per spaCy Doc |
| **Tag Layer** | WebAnno layer name containing POS tags |
| **Lemma Layer** | WebAnno layer name containing lemmas |
| **Shuffle Mode** | How to randomize data before splitting |
| **Random Seed** | For reproducible shuffling |
| **Transliteration** | Convert between Cyrillic and Latin scripts |

### GUI Screenshot Layout

```
┌─────────────────────────────────────────────────────────┐
│  WebAnno TSV → spaCy DocBin                             │
├─────────────────────────────────────────────────────────┤
│  Input: [Folder ▼] [________________________] [Browse]  │
│  Output Folder:    [________________________] [Browse]  │
│  Output Root:      [dataset_______________]             │
├─────────────────────────────────────────────────────────┤
│  Model:            [blank:sr______________]             │
│  Sentences/Doc:    [3_____]                             │
│  Tag Layer:        [coarseValue___________]             │
│  Lemma Layer:      [value_4_______________]             │
├─────────────────────────────────────────────────────────┤
│  Shuffle Mode:     [chunk ▼]  Seed: [______]            │
│  Transliteration:  [none ▼]   Lang: [sr ▼]              │
├─────────────────────────────────────────────────────────┤
│  [        🚀 Convert        ]                           │
├─────────────────────────────────────────────────────────┤
│  Log Output:                                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Parsing file1.tsv...                              │  │
│  │ Parsed 150 sentences                              │  │
│  │ Converting train split...                         │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## ⌨️ CLI (Command-Line Interface)

### Basic Usage

```bash
webanno-to-spacy INPUT OUTPUT [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT` | Folder with TSV files or semicolon-separated file paths |
| `OUTPUT` | Output directory for `.spacy` files |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-r, --root` | `dataset` | Root name for output files |
| `-m, --model` | `blank:sr` | spaCy model name |
| `-n, --sentences-per-doc` | `3` | Sentences per Doc |
| `--tag-layer` | `coarseValue` | POS tag layer name |
| `--lemma-layer` | `value_4` | Lemma layer name |
| `--shuffle-mode` | `chunk` | `chunk`, `sentence`, or `none` |
| `--seed` | (random) | Random seed for reproducibility |
| `-v, --verbose` | off | Enable debug logging |

### Examples

```bash
# Basic conversion
webanno-to-spacy ./tsv_files ./output

# English data with custom name
webanno-to-spacy ./data ./output -m blank:en -r my_corpus

# Reproducible shuffle with seed
webanno-to-spacy ./data ./output --shuffle-mode sentence --seed 42

# No shuffling (preserve order)
webanno-to-spacy ./data ./output --shuffle-mode none

# Verbose output
webanno-to-spacy ./data ./output -v

# Custom layer names
webanno-to-spacy ./data ./output --tag-layer pos --lemma-layer lemma
```

### Output Files

The CLI creates these files:
```
output/
├── dataset-train.spacy   # 80% of data
├── dataset-dev.spacy     # 10% of data
├── dataset-eval.spacy    # Same as dev (alias)
├── dataset-test.spacy    # 10% of data
└── dataset-all.spacy     # All data combined
```

---

## 🐍 Python API

### Parsing WebAnno TSV

```python
from webanno_spacy_converter import WebAnnoNELParser, WebAnnoLEXISParser

# Standard NEL format
parser = WebAnnoNELParser("file.tsv")
parser.parse()

# Access parsed data
for sentence in parser.sentences:
    print(f"Text: {sentence.text}")
    print(f"Tokens: {[t.text for t in sentence.tokens]}")
    
    # Entities: (start_char, end_char, label, link)
    for start, end, label, link in sentence.entities:
        entity_text = sentence.text[start:end]
        print(f"  Entity: {entity_text} [{label}] -> {link}")

# LEXIS format (with Multi-Word Expressions)
lexis_parser = WebAnnoLEXISParser("lexis_file.tsv")
lexis_parser.parse()
```

### Converting to spaCy

```python
from webanno_spacy_converter import AnnotationSentencesToDocBinConverterV2
import spacy

nlp = spacy.blank("sr")
converter = AnnotationSentencesToDocBinConverterV2(
    nlp,
    sentences_per_doc=3,      # Combine 3 sentences per Doc
    tag_layer="coarseValue",  # POS tag layer name
    lemma_layer="value_4",    # Lemma layer name
    clean_hidden_chars=True   # Remove zero-width characters
)

# Convert sentences to DocBin
docbin = converter.convert(sentences)
docbin.to_disk("output.spacy")

# Load and inspect
docbin_loaded = DocBin().from_disk("output.spacy")
docs = list(docbin_loaded.get_docs(nlp.vocab))

for doc in docs:
    print(f"Text: {doc.text}")
    for ent in doc.ents:
        print(f"  {ent.text} [{ent.label_}] -> {ent.kb_id_}")
```

### Converting spaCy to WebAnno

```python
from webanno_spacy_converter import DocBinToAnnotationSentencesConverter, WebAnnoNELWriter
import spacy
from spacy.tokens import DocBin

# Load spaCy docs
nlp = spacy.blank("sr")
docbin = DocBin().from_disk("data.spacy")
docs = list(docbin.get_docs(nlp.vocab))

# Convert to annotation sentences
converter = DocBinToAnnotationSentencesConverter()
sentences = converter.convert(docs)

# Write to WebAnno TSV
writer = WebAnnoNELWriter(sentences)
writer.save("output.tsv")
```

### Working with Entities

```python
from webanno_spacy_converter import AnnotationSentence, AnnotationToken

# Create tokens
tokens = [
    AnnotationToken(text="Nikola", start=0, end=6),
    AnnotationToken(text="Tesla", start=7, end=12),
    AnnotationToken(text="was", start=13, end=16),
    AnnotationToken(text="born", start=17, end=21),
    AnnotationToken(text="in", start=22, end=24),
    AnnotationToken(text="Smiljan", start=25, end=32),
]

# Create sentence with entities
sentence = AnnotationSentence(
    text="Nikola Tesla was born in Smiljan",
    tokens=tokens,
    entities=[
        (0, 12, "PERSON", "Q9036"),      # Nikola Tesla -> Wikidata Q9036
        (25, 32, "LOC", "Q623728"),       # Smiljan -> Wikidata Q623728
    ]
)
```

---

## 📄 WebAnno TSV Format

### TSV 3.3 Structure

```tsv
#FORMAT=WebAnno TSV 3.3
#T_SP=custom.Span|value|identifier


#Text=Nikola Tesla was born in Smiljan.
1-1	0-6	Nikola	PERSON[1]	Q9036[1]
1-2	7-12	Tesla	PERSON[1]	Q9036[1]
1-3	13-16	was	_	_
1-4	17-21	born	_	_
1-5	22-24	in	_	_
1-6	25-32	Smiljan	LOC	Q623728
1-7	32-33	.	_	_

#Text=He invented AC power.
2-1	0-2	He	_	_
2-2	3-11	invented	_	_
2-3	12-14	AC	_	_
2-4	15-20	power	_	_
2-5	20-21	.	_	_
```

### Column Meanings

| Column | Example | Description |
|--------|---------|-------------|
| 1 | `1-1` | Sentence-Token ID |
| 2 | `0-6` | Character offsets (start-end) |
| 3 | `Nikola` | Token text |
| 4+ | `PERSON[1]` | Annotation layers (NER, POS, etc.) |

### Multi-Token Entities

Entities spanning multiple tokens use `[N]` suffixes:
```tsv
1-1	0-6	Nikola	PERSON[1]	Q9036[1]
1-2	7-12	Tesla	PERSON[1]	Q9036[1]
```
Both tokens share `[1]`, indicating they form one entity.

---

## ⚙️ Configuration Options

### Shuffle Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `chunk` | Shuffle document chunks, preserve file boundaries | Default, balanced approach |
| `sentence` | Shuffle individual sentences | Maximum randomization |
| `none` | Preserve original order | When order matters |

### Sentences Per Doc

Controls how many sentences are combined into each spaCy `Doc`:

| Value | Effect |
|-------|--------|
| `1` | One sentence per Doc (simplest) |
| `3` | Three sentences per Doc (default, good for context) |
| `5+` | More sentences (good for document-level features) |

### Layer Names

Common WebAnno layer configurations:

| Annotation Type | Common Layer Names |
|-----------------|-------------------|
| POS tags | `coarseValue`, `pos`, `POS`, `xpos` |
| Lemmas | `value_4`, `lemma`, `Lemma` |
| NER | `value`, `ner`, `NER` |
| NEL | `identifier`, `link`, `kb_id` |

---

## 🔤 Transliteration

Convert between Cyrillic and Latin scripts using the built-in transliteration feature.

### Supported Languages

| Code | Language |
|------|----------|
| `sr` | Serbian |
| `me` | Montenegrin |
| `mk` | Macedonian |
| `ru` | Russian |
| `bg` | Bulgarian |
| `ua` | Ukrainian |
| `by` | Belarusian |

### GUI Usage

1. Select **Transliteration** mode: `to_latin` or `to_cyrillic`
2. Select the appropriate **Language**
3. Convert as usual

### What Gets Transliterated

| Content | Transliterated? |
|---------|-----------------|
| Sentence text | ✅ Yes |
| Token text | ✅ Yes |
| Lemmas | ✅ Yes |
| Entity labels (PERSON, LOC) | ❌ No (codes preserved) |
| POS tags (NOUN, VERB) | ❌ No (codes preserved) |
| Wikidata QIDs (Q9036) | ❌ No (identifiers preserved) |

### Python API

```python
import cyrtranslit

# Convert text
latin = cyrtranslit.to_latin("Никола Тесла", "sr")
# → "Nikola Tesla"

cyrillic = cyrtranslit.to_cyrillic("Nikola Tesla", "sr")
# → "Никола Тесла"
```

---

## 📂 Project Structure

```
webanno_spacy_converter/
├── __init__.py           # Package exports and version
├── cli.py                # Command-line interface
├── gui.py                # Tkinter GUI
├── config.py             # Configuration constants
│
├── parsers/              # TSV parsing
│   ├── base_parser.py    # Base parser class
│   └── tsv_parser_v3.py  # WebAnno TSV 3.3 parser
│
├── converters/           # Format conversion
│   ├── webanno_to_spacy.py  # TSV → spaCy DocBin
│   └── spacy_to_webanno.py  # spaCy → TSV
│
├── writers/              # Output writers
│   └── webanno_writer.py # Write WebAnno TSV
│
├── models/               # Data structures
│   ├── annotation_token.py     # Token class
│   ├── annotation_sentence.py  # Sentence class
│   └── sentence_with_mwes.py   # MWE support
│
└── utils/                # Utilities
    ├── chunking.py       # Dataset chunking
    ├── file_io.py        # File operations
    └── text_cleaning.py  # Hidden char removal
```

---

## 📚 API Reference

### Main Classes

#### `WebAnnoNELParser`
```python
WebAnnoNELParser(file_path: str)
```
Parse WebAnno TSV files with NER/NEL annotations.

**Attributes:**
- `sentences`: List of `AnnotationSentence` objects
- `layers`: Dictionary of layer names and types

**Methods:**
- `parse()`: Parse the file and populate `sentences`

#### `WebAnnoLEXISParser`
```python
WebAnnoLEXISParser(file_path: str)
```
Parse LEXIS-format TSV with Multi-Word Expression support.

#### `AnnotationSentencesToDocBinConverterV2`
```python
AnnotationSentencesToDocBinConverterV2(
    nlp: spacy.Language,
    sentences_per_doc: int = 3,
    tag_layer: str = "coarseValue",
    lemma_layer: str = "value_4",
    clean_hidden_chars: bool = True
)
```
Convert parsed sentences to spaCy DocBin format.

**Methods:**
- `convert(sentences) -> DocBin`: Convert sentences to DocBin

#### `WebAnnoNELWriter`
```python
WebAnnoNELWriter(sentences: List[AnnotationSentence])
```
Write sentences back to WebAnno TSV format.

**Methods:**
- `save(file_path: str)`: Save to TSV file
- `to_string() -> str`: Get TSV as string

### Data Classes

#### `AnnotationToken`
```python
AnnotationToken(
    text: str,
    start: int,      # Character offset start
    end: int,        # Character offset end
    layers: dict = {}  # Annotation layers
)
```

#### `AnnotationSentence`
```python
AnnotationSentence(
    text: str,
    tokens: List[AnnotationToken],
    entities: List[Tuple[int, int, str, str]] = []
    # entities: (start, end, label, link)
)
```

---

## 🔧 Troubleshooting

### Common Issues

#### "No sentences parsed"
- Check that input files are valid WebAnno TSV 3.3 format
- Verify files have `.tsv` extension
- Check file encoding (should be UTF-8)

#### "Entity alignment error"
- Entity character offsets don't match text
- Run with `-v` flag to see detailed debug output
- Check for hidden Unicode characters in source text

#### "Import error: cyrtranslit"
```bash
pip install cyrtranslit
```

#### "spaCy model not found"
```bash
# For blank models, use blank:lang format
python -c "import spacy; nlp = spacy.blank('sr')"

# For trained models, download first
python -m spacy download en_core_web_sm
```

### Hidden Characters

The converter automatically removes problematic hidden characters:
- Zero-width space (U+200B)
- Zero-width non-joiner (U+200C)
- Zero-width joiner (U+200D)
- Word joiner (U+2060)
- Soft hyphen (U+00AD)
- Byte order mark (U+FEFF)

To disable: set `clean_hidden_chars=False` in converter.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `pytest tests/ -v`
5. Submit a pull request

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_parsers.py -v

# Run with coverage
pytest tests/ --cov=webanno_spacy_converter
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

- **Issues**: [GitHub Issues](https://github.com/te-sla/webanno-spacy-converter/issues)
- **Repository**: [GitHub](https://github.com/te-sla/webanno-spacy-converter)
