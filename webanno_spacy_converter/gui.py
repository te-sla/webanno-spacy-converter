import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from typing import List
import random
import traceback
import logging
from datetime import datetime

import spacy
from spacy.tokens import DocBin

from .parsers.tsv_parser_v3 import WebAnnoNELParser
from .converters.webanno_to_spacy import AnnotationSentencesToDocBinConverterV2

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ErrorDialog:
    """A dialog for displaying errors with copy and save options."""
    
    def __init__(self, parent, title: str, error_text: str):
        self.error_text = error_text
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("800x600")
        self.dialog.minsize(600, 400)
        
        # Make it modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Main frame
        main_frame = tk.Frame(self.dialog, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Error icon and title
        title_frame = tk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(title_frame, text="❌", font=("Arial", 24)).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(title_frame, text=title, font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        
        # Scrolled text widget for error content
        self.text_widget = scrolledtext.ScrolledText(
            main_frame, 
            wrap=tk.WORD, 
            font=("Consolas", 10),
            bg="#2d2d2d",
            fg="#ffffff",
            insertbackground="#ffffff"
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.text_widget.insert(tk.END, error_text)
        self.text_widget.config(state=tk.DISABLED)  # Read-only but selectable
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        tk.Button(
            button_frame, 
            text="📋 Copy to Clipboard", 
            command=self._copy_to_clipboard,
            width=18
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(
            button_frame, 
            text="💾 Save to File", 
            command=self._save_to_file,
            width=18
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        tk.Button(
            button_frame, 
            text="Close", 
            command=self.dialog.destroy,
            width=10
        ).pack(side=tk.RIGHT)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="", fg="green")
        self.status_label.pack(fill=tk.X, pady=(5, 0))
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")
        
        # Focus on dialog
        self.dialog.focus_set()
    
    def _copy_to_clipboard(self):
        """Copy error text to clipboard."""
        self.dialog.clipboard_clear()
        self.dialog.clipboard_append(self.error_text)
        self.status_label.config(text="✓ Copied to clipboard!", fg="green")
        self.dialog.after(2000, lambda: self.status_label.config(text=""))
    
    def _save_to_file(self):
        """Save error text to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"conversion_error_{timestamp}.txt"
        
        filepath = filedialog.asksaveasfilename(
            parent=self.dialog,
            title="Save Error Log",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.error_text)
                self.status_label.config(text=f"✓ Saved to {os.path.basename(filepath)}", fg="green")
            except Exception as e:
                self.status_label.config(text=f"✗ Save failed: {e}", fg="red")


class WebAnnoToSpacyGUI:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("WebAnno TSV → spaCy DocBin")

        # State
        self.input_mode = tk.StringVar(value="folder")  # "folder" or "files"
        self.input_path = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.output_root = tk.StringVar(value="dataset")
        self.sentences_per_doc = tk.IntVar(value=3)
        self.model_name = tk.StringVar(value="blank:sr")
        # Default layer names for ELEXIS WebAnno TSV
        self.tag_layer = tk.StringVar(value="coarseValue")   # POS
        self.lemma_layer = tk.StringVar(value="value_4")     # lemma

        # Build UI
        self._build_ui()

    def _build_ui(self) -> None:
        frame = tk.Frame(self.master, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Input mode
        mode_frame = tk.LabelFrame(frame, text="Input mode")
        mode_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Radiobutton(
            mode_frame,
            text="Folder with TSV files",
            variable=self.input_mode,
            value="folder",
            command=self._update_input_controls,
        ).pack(anchor=tk.W)
        tk.Radiobutton(
            mode_frame,
            text="Select TSV files",
            variable=self.input_mode,
            value="files",
            command=self._update_input_controls,
        ).pack(anchor=tk.W)

        # Input path
        input_frame = tk.LabelFrame(frame, text="Input")
        input_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Entry(input_frame, textvariable=self.input_path, width=50).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4), pady=4
        )
        tk.Button(input_frame, text="Browse", command=self._browse_input).pack(
            side=tk.LEFT, padx=(0, 4), pady=4
        )

        # Output folder
        out_frame = tk.LabelFrame(frame, text="Output")
        out_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(out_frame, text="Folder:").pack(side=tk.LEFT, padx=(4, 4))
        tk.Entry(out_frame, textvariable=self.output_folder, width=40).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4), pady=4
        )
        tk.Button(out_frame, text="Browse", command=self._browse_output_folder).pack(
            side=tk.LEFT, padx=(0, 4), pady=4
        )

        # Root name, model, sentences per doc, and tag/lemma layers
        options_frame = tk.Frame(frame)
        options_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(options_frame, text="Output root name:").grid(row=0, column=0, sticky=tk.W, padx=(4, 4), pady=2)
        tk.Entry(options_frame, textvariable=self.output_root, width=20).grid(
            row=0, column=1, sticky=tk.W, padx=(0, 8), pady=2
        )

        tk.Label(options_frame, text="Model (path or blank:lang):").grid(row=0, column=2, sticky=tk.W, padx=(4, 4), pady=2)
        tk.Entry(options_frame, textvariable=self.model_name, width=20).grid(
            row=0, column=3, sticky=tk.W, padx=(0, 4), pady=2
        )

        tk.Label(options_frame, text="Sentences per Doc:").grid(row=1, column=0, sticky=tk.W, padx=(4, 4), pady=2)
        tk.Spinbox(options_frame, from_=1, to=1000, textvariable=self.sentences_per_doc, width=6).grid(
            row=1, column=1, sticky=tk.W, padx=(0, 8), pady=2
        )

        tk.Label(options_frame, text="POS layer name:").grid(row=1, column=2, sticky=tk.W, padx=(4, 4), pady=2)
        tk.Entry(options_frame, textvariable=self.tag_layer, width=20).grid(
            row=1, column=3, sticky=tk.W, padx=(0, 4), pady=2
        )

        tk.Label(options_frame, text="Lemma layer name:").grid(row=2, column=0, sticky=tk.W, padx=(4, 4), pady=2)
        tk.Entry(options_frame, textvariable=self.lemma_layer, width=20).grid(
            row=2, column=1, sticky=tk.W, padx=(0, 8), pady=2
        )

        # spacer to align grid
        options_frame.grid_columnconfigure(4, weight=1)

        # Run button
        run_button = tk.Button(frame, text="Convert", command=self._run_conversion)
        run_button.pack(pady=(4, 0))

        self._update_input_controls()

    def _show_error_dialog(self, title: str, error_text: str):
        """Show a detailed error dialog with copy and save options."""
        ErrorDialog(self.master, title, error_text)

    def _update_input_controls(self) -> None:
        # Currently nothing dynamic beyond browse behavior, but hook kept for future
        pass

    def _browse_input(self) -> None:
        if self.input_mode.get() == "folder":
            path = filedialog.askdirectory(title="Select folder with TSV files")
            if path:
                self.input_path.set(path)
        else:
            files = filedialog.askopenfilenames(
                title="Select TSV files",
                filetypes=[("TSV files", "*.tsv"), ("All files", "*.*")],
            )
            if files:
                # Store as OS-path-separated list
                self.input_path.set(";".join(files))

    def _browse_output_folder(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_folder.set(path)

    def _collect_input_files(self) -> List[str]:
        mode = self.input_mode.get()
        raw = self.input_path.get().strip()
        if not raw:
            return []

        if mode == "folder":
            folder = raw
            if not os.path.isdir(folder):
                return []
            return [
                os.path.join(folder, name)
                for name in os.listdir(folder)
                if name.lower().endswith(".tsv")
            ]
        else:
            return [p for p in raw.split(";") if p]

    def _run_conversion(self) -> None:
        input_files = self._collect_input_files()
        if not input_files:
            messagebox.showerror("Error", "No TSV files found/selected.")
            return

        out_dir = self.output_folder.get().strip()
        if not out_dir:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        root = self.output_root.get().strip() or "dataset"
        s_per_doc = max(1, int(self.sentences_per_doc.get() or 10))
        tag_layer = self.tag_layer.get().strip() or None
        lemma_layer = self.lemma_layer.get().strip() or None
        model_name = self.model_name.get().strip() or "blank:sr"

        try:
            if model_name.startswith("blank:"):
                lang = model_name.split(":", 1)[1].strip() or "sr"
                nlp = spacy.blank(lang)
            else:
                nlp = spacy.load(model_name)
        except Exception as e:  # pragma: no cover - GUI environment specific
            messagebox.showerror("Error", f"Failed to load spaCy pipeline '{model_name}': {e}")
            return

        converter = AnnotationSentencesToDocBinConverterV2(
            nlp,
            sentences_per_doc=s_per_doc,
            tag_layer=tag_layer,
            lemma_layer=lemma_layer,
            ner=True,
            nel=True,
        )

        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot create output folder: {e}")
            return

        try:
            all_sentences: List = []
            current_file = ""
            
            # Parse all files with detailed error tracking
            for path in input_files:
                current_file = path
                logger.info(f"Parsing file: {path}")
                try:
                    parser = WebAnnoNELParser(path)
                    file_sentences = parser.parse()
                    logger.info(f"  Parsed {len(file_sentences)} sentences from {os.path.basename(path)}")
                    all_sentences.extend(file_sentences)
                except Exception as e:
                    error_msg = f"Error parsing file: {path}\n\nError: {e}\n\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    self._show_error_dialog("Parse Error", error_msg)
                    return

            # Simple split: 80% train, 10% dev, 10% test
            total = len(all_sentences)
            if total == 0:
                messagebox.showerror("Error", "Parsed 0 sentences from input files.")
                return

            logger.info(f"Total sentences parsed: {total}")

            # Shuffle sentences before splitting so train/dev/test are mixed
            random.shuffle(all_sentences)

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

            for split_name, split_sents in splits.items():
                if not split_sents:
                    continue
                logger.info(f"Converting {split_name} split ({len(split_sents)} sentences)...")
                
                try:
                    docbin: DocBin = converter.convert(split_sents)
                    out_path = os.path.join(out_dir, f"{root}-{split_name}.spacy")
                    docbin.to_disk(out_path)
                    logger.info(f"  Saved to {out_path}")
                except Exception as e:
                    # Try to identify problematic sentence
                    error_msg = self._find_problematic_sentence(converter, split_sents, split_name, e)
                    logger.error(error_msg)
                    self._show_error_dialog("Conversion Error", error_msg)
                    return

            # Optional: one combined DocBin with all sentences for debugging
            logger.info("Converting all sentences...")
            try:
                all_docbin: DocBin = converter.convert(all_sentences)
                all_path = os.path.join(out_dir, f"{root}-all.spacy")
                all_docbin.to_disk(all_path)
                logger.info(f"  Saved to {all_path}")
            except Exception as e:
                error_msg = self._find_problematic_sentence(converter, all_sentences, "all", e)
                logger.error(error_msg)
                self._show_error_dialog("Conversion Error", error_msg)
                return

        except Exception as e:  # pragma: no cover - GUI environment specific
            error_msg = f"Conversion failed: {e}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            self._show_error_dialog("Error", error_msg)
            return

        messagebox.showinfo("Done", f"Conversion finished successfully.\n\nTotal sentences: {total}\nTrain: {len(train_sents)}\nDev: {len(dev_sents)}\nTest: {len(test_sents)}")

    def _find_problematic_sentence(self, converter, sentences, split_name, original_error):
        """Try to identify which sentence caused the conversion error."""
        error_lines = [
            f"Conversion failed for '{split_name}' split",
            f"Original error: {original_error}",
            "",
            "Searching for problematic sentence..."
        ]
        
        # Try converting sentences one by one to find the problematic one
        for i, sent in enumerate(sentences):
            try:
                # Try converting just this one sentence
                converter.convert([sent])
            except Exception as e:
                error_lines.append("")
                error_lines.append(f"FOUND PROBLEMATIC SENTENCE #{i}:")
                error_lines.append(f"  Text: {repr(sent.text[:100] if sent.text else 'None')}...")
                error_lines.append(f"  Tokens: {len(sent.tokens)}")
                if sent.tokens:
                    error_lines.append(f"  First token: {repr(sent.tokens[0].text)} at {sent.tokens[0].start}-{sent.tokens[0].end}")
                    error_lines.append(f"  Last token: {repr(sent.tokens[-1].text)} at {sent.tokens[-1].start}-{sent.tokens[-1].end}")
                error_lines.append(f"  Entities: {len(sent.entities)}")
                error_lines.append(f"  Error: {e}")
                error_lines.append("")
                error_lines.append(f"Full traceback:\n{traceback.format_exc()}")
                break
        else:
            error_lines.append("")
            error_lines.append("Could not identify single problematic sentence.")
            error_lines.append("The error may occur only when combining multiple sentences.")
            error_lines.append(f"\nFull traceback:\n{traceback.format_exc()}")
        
        return "\n".join(error_lines)


def run_gui() -> None:
    root = tk.Tk()
    WebAnnoToSpacyGUI(root)
    root.mainloop()


if __name__ == "__main__":  # pragma: no cover
    run_gui()
