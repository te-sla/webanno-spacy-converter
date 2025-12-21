from setuptools import setup, find_packages

setup(
    name="webanno_spacy_converter",
    version="0.1.2",
    description="Tools to convert between WebAnno TSV and spaCy formats",
    author="SasaP",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.5",
        "cyrtranslit"
    ],
    entry_points={
        "console_scripts": [
            "webanno-to-spacy=webanno_spacy_converter.cli:main",
            "webanno-to-spacy-gui=webanno_spacy_converter.gui:run_gui",
        ],
    },
    python_requires=">=3.7"
)
