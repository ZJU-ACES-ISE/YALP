class Sample:

    def __init__(self, sample_text: str, templae_id: str = None, sample_id: str = None, log_num: int = 1):
        self.sample_text: str = sample_text
        self.template_id: str = templae_id
        self.sample_id: str = sample_id
        self.log_num: int = log_num
