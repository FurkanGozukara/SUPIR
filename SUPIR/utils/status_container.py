from typing import Dict, List, Union, Any

import numpy as np


class StatusContainer:
    """
    This class is used to store the status of the process
    """
    def __init__(self):
        self.prompt: str = ""
        self.image_data: List[MediaData] = []
        self.is_batch: bool = False
        self.is_video: bool = False
        self.video_params: Dict[str, Any] = {}
        self.process_params: Dict[str, Any] = {}
        self.source_video_path = None


class MediaData:
    """
    This class is used to store the media data for each item being processed
    """
    def __init__(self, media_path="", media_type="image"):
        self.media_path: str = media_path
        self.media_data: Union[None, np.ndarray] = None
        self.media_type: str = media_type
        self.caption: str = ""
        self.outputs: List[str] = []
        self.comparison_video: Union[str, None] = None
        self.metadata_list: List[Dict[str, Any]] = {}
