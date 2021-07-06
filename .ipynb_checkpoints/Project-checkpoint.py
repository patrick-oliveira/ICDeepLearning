from dataclasses import dataclass
from pathlib import Path

@dataclass
class Directory:
    name: str
    directory: str

@dataclass
class Project:
    """
        Basic information about the project.
    """
    base:                     Path = Path(__file__).parents[0]
    unicamp_raw:              Path = base.parent / 'BCI_ssvep_dataset'
    unicamp_signals:          Path = base / 'Datasets' / 'Unicamp'
    unicamp_combined_signals: Path = base / 'Datasets' / 'Unicamp' / 'Combined Signals'
    data:                     Path = base / 'Datasets'
    output:                   Path = base / 'Outputs'
    tsinghua:                 Path = data / 'Tsinghua'
    
    # Adaptar para printar todos os diretórios
    