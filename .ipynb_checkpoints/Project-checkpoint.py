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
    base_dir:            Path = Path(__file__).parents[0]
    raw_ssvep_dir:       Path = base_dir.parent / 'BCI_ssvep_dataset'
    data_dir:            Path = base_dir / 'Data'
    base_series_dir:     Path = data_dir / 'originals'
    combined_series_dir: Path = data_dir / 'combined'
    images_dir:          Path = data_dir / 'images'
    output_dir:          Path = base_dir / 'Outputs'
    tsinghua_raw_dir:    Path = data_dir / 'tsinghua' / 'raw' / 'beta'
    tsinghua_cca_dir:     Path = data_dir / 'tsinghua' / 'cca'
    
    # Adaptar para printar todos os diretórios
    