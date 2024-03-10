class StatusContainer:
    # prompt, stage_1_output_image, result_gallery, event_id, fb_score, fb_text, seed, face_gallery, comparison_video
    def __init__(self):
        self.prompt: str = ""
        self.image_data = {}
        self.output_data = {}
        self.events_dict = {}
        self.result_gallery = None
        self.event_id = ""
        self.fb_score = 0
        self.fb_text = ""
        self.seed = 0
        self.face_gallery = None
        self.comparison_video = None
        self.llava_caption = ""
        self.llava_captions = []
