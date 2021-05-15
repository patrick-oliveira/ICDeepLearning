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
    data:            Path = base / 'Data'
    output:          Path = base / 'Outputs'
    tsinghua:        Path = base / 'Tsinghua'
    
    # Adaptar para printar todos os diretórios
    