from dataclasses import dataclass
from pathlib import Path

@dataclass
class Project:
    """
    This class represents our project. It stores useful information about the structure,
    e.g. patchs
    """
    base_dir: Path = Path(__file__).parents[2]
    data_dir = base_dir / 'Data'
    output_dir = base_dir / 'Outputs'
    script_dir = base_dir / 'Scripts'
    
    def __post_init__(self):
        # create the directory if they does not exist
        self.data_dir.mkdir(exist_ok = True)
        