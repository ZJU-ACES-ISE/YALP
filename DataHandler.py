from typing import List, Optional

from Snowflake import Snowflake
from Sample import Sample
from Template import Template


class DataHandler:
    def __init__(self):
        self.snowflake = Snowflake(worker_id=1, datacenter_id=1)
        self.templates: List[Template] = []
        self.samples: List[Sample] = []

    def select_sample_by_id(self, sample_id: str) -> Sample:
        return next(filter(lambda sample: sample.sample_id == sample_id, self.samples))

    def select_samples(self) -> List[Sample]:
        return self.samples

    def select_template_by_id(self, template_id: str) -> Optional[Template]:
        return next(filter(lambda template: template.template_id == template_id, self.templates), None)

    def select_samples_by_template_id(self, template_id) -> List[Sample]:
        return list(filter(lambda sample: sample.template_id == template_id, self.samples))

    def update_template_id_of_sample_by_template(self, new_template: Template, origin_template: Template) -> None:
        samples = self.select_samples_by_template_id(origin_template.template_id)
        for sample in samples:
            sample.template_id = new_template.template_id

    def select_sample_len_by_template_id(self, template_id: str) -> int:
        samples = self.select_samples_by_template_id(template_id)
        return sum([sample.log_num for sample in samples])

    def insert_template(self, template_without_id: Template) -> Template:
        template_without_id.template_id = self.snowflake.generate_id()
        self.templates.append(template_without_id)
        return template_without_id

    def empty_template(self, template: Template) -> None:
        template.template_text = '<EMPTY>'

    def update_template_id_of_sample(self, new_template: Template, sample: Sample) -> Sample:
        return sample

    def remove_template(self, template: Template) -> None:
        self.templates.remove(template)

    def select_templates(self) -> List[Template]:
        return self.templates

    # def Sample
    def select_sample_by_text(self, sample_text: str) -> Optional[Sample]:
        return next(filter(lambda sample: sample.sample_text == sample_text, self.samples), None)

    def update_log_num_of_sample(self, sample: Sample) -> Sample:
        return sample

    def insert_sample(self, sample_without_id: Sample) -> Sample:
        sample_without_id.sample_id = self.snowflake.generate_id()
        self.samples.append(sample_without_id)
        return sample_without_id

# if __name__ == '__main__':
#     # dbh = InMemoryHandler()
#     # dbh.insert_sample(Sample('0', '-'))
#     # dbh.insert_sample(Sample('1', '-'))
#     # dbh.insert_sample(Sample('2', '-'))
#     #
#     # print([sample.__dict__ for sample in dbh.samples])
#     #
#     # sample_2 = dbh.select_sample_by_text('2')
#     # sample_2.log_num += 1
#     # print([sample.__dict__ for sample in dbh.samples])
#
#     dbh = InMemoryHandler()
#     dbh.insert_template(Template('0'))
#     dbh.insert_template(Template('1'))
#     dbh.insert_template(Template('2'))
#     print([template.__dict__ for template in dbh.templates])
#
#     dbh.remove_template(dbh.templates[1])
#     print([template.__dict__ for template in dbh.templates])
