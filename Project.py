from dataclasses import dataclass
from pathlib import Path


@dataclass
class Project:
    """
        Basic information about the project.
    """
    base_dir: Path = Path(__file__).parents[0]
    raw_ssvep_dir = base_dir.parent / 'BCI_ssvep_dataset'
    data_dir = base_dir / 'Data'
    base_series_dir = data_dir / 'originals'
    combined_series_dir = data_dir / 'combined'
    images_dir = data_dir / 'images'
    output_dir = base_dir / 'Outputs'
    