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
    base:            Path = Path(__file__).parents[0]
    data:            Path = base / 'Datasets'
    output:          Path = base / 'Outputs'
    tsinghua:        Path = data / 'Tsinghua'
    
    # Adaptar para printar todos os diretórios
    