"""
Command-line interface for WebAnno TSV to spaCy DocBin conversion.
"""

import argparse
import logging
import os
import random
import sys
import traceback
from typing import List, Dict, Optional

import spacy
from spacy.tokens import DocBin

from .parsers.tsv_parser_v3 import WebAnnoNELParser
from .converters.webanno_to_spacy import AnnotationSentencesToDocBinConverterV2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_input_files(input_path: str) -> List[str]:
    """
    Collect TSV files from input path.
    
    Args:
        input_path: Either a folder path or semicolon-separated file paths
        
    Returns:
        List of TSV file paths
    """
    if os.path.isdir(input_path):
        return [
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if name.lower().endswith(".tsv")
        ]
    else:
        # Assume semicolon-separated file paths
        return [p.strip() for p in input_path.split(";") if p.strip()]


def create_chunks(
    sentences_by_file: Dict[str, List], 
    chunk_size: int
) -> List[List]:
    """
    Group sentences into chunks, respecting file boundaries.
    Each file's sentences are split into chunks of chunk_size.
    
    Args:
        sentences_by_file: Dict mapping file path to list of sentences
        chunk_size: Maximum sentences per chunk
        
    Returns:
        List of chunks (each chunk is a list of sentences)
    """
    chunks = []
    # Sort files alphabetically for consistent ordering
    for file_path in sorted(sentences_by_file.keys()):
        file_sents = sentences_by_file[file_path]
        # Split this file's sentences into chunks
        for i in range(0, len(file_sents), chunk_size):
            chunk = file_sents[i:i + chunk_size]
            chunks.append(chunk)
    return chunks


def prepare_sentences(
    sentences_by_file: Dict[str, List],
    mode: str,
    chunk_size: int,
    seed: Optional[int]
) -> List:
    """
    Prepare sentences for train/dev/test split based on shuffle mode.
    
    Args:
        sentences_by_file: Dict mapping file path to list of sentences
        mode: "chunk" (shuffle chunks, preserve file boundaries), 
              "sentence" (shuffle individual sentences),
              "none" (preserve original order)
        chunk_size: Number of sentences per chunk (used for "chunk" mode)
        seed: Random seed for reproducibility (None = random)
    
    Returns:
        Flattened list of sentences in the appropriate order
    """
    # Set seed if provided
    if seed is not None:
        random.seed(seed)
        logger.info(f"Using random seed: {seed}")
    
    if mode == "chunk":
        # Create chunks respecting file boundaries, then shuffle chunks
        chunks = create_chunks(sentences_by_file, chunk_size)
        random.shuffle(chunks)
        # Flatten chunks back to sentence list
        return [sent for chunk in chunks for sent in chunk]
    
    elif mode == "sentence":
        # Flatten all sentences then shuffle individually
        all_sents = []
        for file_path in sorted(sentences_by_file.keys()):
            all_sents.extend(sentences_by_file[file_path])
        random.shuffle(all_sents)
        return all_sents
    
    else:  # mode == "none"
        # Preserve order: alphabetical by file, original order within file
        all_sents = []
        for file_path in sorted(sentences_by_file.keys()):
            all_sents.extend(sentences_by_file[file_path])
        return all_sents


def convert(
    input_path: str,
    output_dir: str,
    output_root: str = "dataset",
    model_name: str = "blank:sr",
    sentences_per_doc: int = 3,
    tag_layer: Optional[str] = "coarseValue",
    lemma_layer: Optional[str] = "value_4",
    shuffle_mode: str = "chunk",
    seed: Optional[int] = None,
) -> bool:
    """
    Convert WebAnno TSV files to spaCy DocBin format.
    
    Args:
        input_path: Path to folder with TSV files or semicolon-separated file paths
        output_dir: Output directory for .spacy files
        output_root: Root name for output files (e.g., "dataset" -> "dataset-train.spacy")
        model_name: spaCy model name or "blank:lang" for blank model
        sentences_per_doc: Number of sentences to combine into each Doc
        tag_layer: Layer name for POS tags (None to skip)
        lemma_layer: Layer name for lemmas (None to skip)
        shuffle_mode: "chunk" (default), "sentence", or "none"
        seed: Random seed for reproducibility (None = random)
        
    Returns:
        True if conversion succeeded, False otherwise
    """
    # Collect input files
    input_files = collect_input_files(input_path)
    if not input_files:
        logger.error("No TSV files found/selected.")
        return False
    
    logger.info(f"Found {len(input_files)} TSV files")
    
    # Load spaCy model
    try:
        if model_name.startswith("blank:"):
            lang = model_name.split(":", 1)[1].strip() or "sr"
            nlp = spacy.blank(lang)
            logger.info(f"Created blank spaCy model for language: {lang}")
        else:
            nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load spaCy pipeline '{model_name}': {e}")
        return False
    
    # Create converter
    converter = AnnotationSentencesToDocBinConverterV2(
        nlp,
        sentences_per_doc=sentences_per_doc,
        tag_layer=tag_layer,
        lemma_layer=lemma_layer,
        ner=True,
        nel=True,
    )
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Cannot create output folder: {e}")
        return False
    
    # Parse all files
    sentences_by_file: Dict[str, List] = {}
    
    for path in input_files:
        logger.info(f"Parsing file: {path}")
        try:
            parser = WebAnnoNELParser(path)
            file_sentences = parser.parse()
            logger.info(f"  Parsed {len(file_sentences)} sentences from {os.path.basename(path)}")
            sentences_by_file[path] = file_sentences
        except Exception as e:
            logger.error(f"Error parsing file: {path}\n\nError: {e}\n\n{traceback.format_exc()}")
            return False
    
    # Count total sentences
    total = sum(len(sents) for sents in sentences_by_file.values())
    if total == 0:
        logger.error("Parsed 0 sentences from input files.")
        return False
    
    logger.info(f"Total sentences parsed: {total} from {len(sentences_by_file)} files")
    
    # Log shuffle mode for transparency
    mode_descriptions = {
        "chunk": "chunk (preserves file boundaries, shuffles document chunks)",
        "sentence": "sentence (shuffles individual sentences, breaks context)",
        "none": "none (preserves original order)"
    }
    logger.info(f"Using shuffle mode: {mode_descriptions.get(shuffle_mode, shuffle_mode)}")
    
    # Prepare sentences based on shuffle mode
    all_sentences = prepare_sentences(sentences_by_file, shuffle_mode, sentences_per_doc, seed)
    
    # Simple split: 80% train, 10% dev, 10% test
    train_end = int(total * 0.8)
    dev_end = int(total * 0.9)
    
    train_sents = all_sentences[:train_end]
    dev_sents = all_sentences[train_end:dev_end]
    test_sents = all_sentences[dev_end:]
    
    splits = {
        "train": train_sents,
        "dev": dev_sents,
        "eval": dev_sents,  # alias
        "test": test_sents,
    }
    
    # Convert and save each split
    for split_name, split_sents in splits.items():
        if not split_sents:
            continue
        logger.info(f"Converting {split_name} split ({len(split_sents)} sentences)...")
        
        try:
            docbin: DocBin = converter.convert(split_sents)
            out_path = os.path.join(output_dir, f"{output_root}-{split_name}.spacy")
            docbin.to_disk(out_path)
            logger.info(f"  Saved to {out_path}")
        except Exception as e:
            logger.error(f"Conversion failed for '{split_name}' split: {e}\n{traceback.format_exc()}")
            return False
    
    # Optional: one combined DocBin with all sentences
    logger.info("Converting all sentences...")
    try:
        all_docbin: DocBin = converter.convert(all_sentences)
        all_path = os.path.join(output_dir, f"{output_root}-all.spacy")
        all_docbin.to_disk(all_path)
        logger.info(f"  Saved to {all_path}")
    except Exception as e:
        logger.error(f"Conversion failed for 'all' split: {e}\n{traceback.format_exc()}")
        return False
    
    logger.info(f"Conversion finished successfully!")
    logger.info(f"  Total sentences: {total}")
    logger.info(f"  Train: {len(train_sents)}")
    logger.info(f"  Dev: {len(dev_sents)}")
    logger.info(f"  Test: {len(test_sents)}")
    
    return True


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Convert WebAnno TSV files to spaCy DocBin format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a folder of TSV files with default settings
  webanno-to-spacy ./tsv_files ./output
  
  # Specify model and output root name
  webanno-to-spacy ./tsv_files ./output -m blank:en -r my_dataset
  
  # Use sentence-level shuffle with a seed for reproducibility
  webanno-to-spacy ./tsv_files ./output --shuffle-mode sentence --seed 42
  
  # Preserve original order (no shuffling)
  webanno-to-spacy ./tsv_files ./output --shuffle-mode none
        """
    )
    
    # Required arguments
    parser.add_argument(
        "input",
        help="Input folder with TSV files or semicolon-separated file paths"
    )
    parser.add_argument(
        "output",
        help="Output directory for .spacy files"
    )
    
    # Optional arguments
    parser.add_argument(
        "-r", "--root",
        default="dataset",
        help="Root name for output files (default: dataset)"
    )
    parser.add_argument(
        "-m", "--model",
        default="blank:sr",
        help="spaCy model name or 'blank:lang' for blank model (default: blank:sr)"
    )
    parser.add_argument(
        "-n", "--sentences-per-doc",
        type=int,
        default=3,
        help="Number of sentences to combine per Doc (default: 3)"
    )
    parser.add_argument(
        "--tag-layer",
        default="coarseValue",
        help="Layer name for POS tags (default: coarseValue)"
    )
    parser.add_argument(
        "--lemma-layer",
        default="value_4",
        help="Layer name for lemmas (default: value_4)"
    )
    parser.add_argument(
        "--shuffle-mode",
        choices=["chunk", "sentence", "none"],
        default="chunk",
        help="Shuffle mode: 'chunk' (default, preserves file boundaries), "
             "'sentence' (individual sentence shuffle), 'none' (preserve order)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible shuffling (default: None = random)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run conversion
    success = convert(
        input_path=args.input,
        output_dir=args.output,
        output_root=args.root,
        model_name=args.model,
        sentences_per_doc=args.sentences_per_doc,
        tag_layer=args.tag_layer,
        lemma_layer=args.lemma_layer,
        shuffle_mode=args.shuffle_mode,
        seed=args.seed,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
