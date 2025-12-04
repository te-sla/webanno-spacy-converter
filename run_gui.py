"""Convenience launcher for the WebAnno → spaCy GUI.

Always run this with the local virtual environment, e.g.::

    .venv\Scripts\python.exe run_gui.py
"""

from webanno_spacy_converter.gui import run_gui


if __name__ == "__main__":
    run_gui()
