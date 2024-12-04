#
#   Copyright 2024 Hopsworks AB
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, List, Union

from jinja2 import Environment, PackageLoader, Template, select_autoescape


if TYPE_CHECKING:
    from hsml.deployment_schema import DeploymentSchema
    from hsml.model_schema import ModelSchema


class CodeTemplates(Enum):
    # Enum mapping the template to a file name
    PREDICTOR = "predictor.j2"


class CodeGenerationEngine:
    def __init__(self) -> None:
        self.environment = Environment(
            loader=PackageLoader("hopsworks.hsfs"),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def load_template(self, code_template_name: Union[str, CodeTemplates]) -> Template:
        if isinstance(code_template_name, CodeTemplates):
            return self.environment.get_template(code_template_name.value)
        else:
            return self.environment.get_template(code_template_name)

    def generate_predictor(
        self,
        enable_logging,
        deployment_schema: DeploymentSchema,
        model_schema: ModelSchema,
        training_dataset_feature_names: List[str],
    ) -> str:
        """
        Function that generates a predictor file and returns the path the file.
        """
        template = self.load_template(CodeTemplates.PREDICTOR)
        return template.render(
            async_logger=enable_logging,
            deployemnt_schema=deployment_schema,
            model_schema=model_schema,
            feature_view_features=training_dataset_feature_names,
        )
