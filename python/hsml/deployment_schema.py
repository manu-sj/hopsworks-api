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

import json
from typing import Dict, List, Optional

from hsml.schema import Schema


class DeploymentSchema:
    """Create a schema for a deployment.

    # Arguments
        serving_key_schema: Schema to describe the serving key required for the deployment.
        passed_feature_schema: Schema to describe the features passed directly to the deployment.
        request_parameter_schema: Schema to describe the request parameters required to fetch feature vector for the deployment.
        output_schema: Schema to describe the output from the deployment.

    # Returns
        `DeploymentSchema`. The deployment schema object.
    """

    def __init__(
        self,
        serving_key_schema: Optional[Schema] = None,
        passed_feature_schema: Optional[Schema] = None,
        request_parameter_schema: Optional[Schema] = None,
        output_schema: Optional[Schema] = None,
        **kwargs,
    ):
        self.serving_key_schema = serving_key_schema

        self.passed_feature_schema = passed_feature_schema

        self.request_parameter_schema = request_parameter_schema

        self.output_schema = output_schema

    def json(self) -> str:
        return json.dumps(
            self, default=lambda o: getattr(o, "__dict__", o), sort_keys=True, indent=2
        )

    def to_dict(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Get dict representation of the ModelSchema.
        """
        return json.loads(self.json())

    @property
    def passed_features(self) -> List[str]:
        """The name of the features that would be passed to the deployment."""
        return (
            sorted([schema.name for schema in self.passed_feature_schema])
            if self.passed_feature_schema
            else []
        )

    @property
    def serving_keys(self) -> List[str]:
        """The name of the feature that are the serving keys required to the retrive feature vectors from the feature store."""
        return (
            sorted([schema.name for schema in self.serving_key_schema])
            if self.serving_key_schema
            else []
        )

    @property
    def request_parameters(self) -> List[str]:
        """The request parameters required for the computing on-demand features present in the feature view."""
        return (
            sorted([schema.name for schema in self.request_parameter_schema])
            if self.request_parameter_schema
            else []
        )

    @property
    def input_features(self) -> List[str]:
        """Inputs features that must be provided to the deployment"""

        return self.serving_keys + self.passed_features + self.request_parameters

    @property
    def output_features(self) -> List[str]:
        """Output features that must be provided to the deployment"""
        return (
            sorted([schema.name for schema in self.output_schema])
            if self.output_schema
            else []
        )

    def __repr__(self):
        return (
            f"DeploymentSchema(inputs: {self.input_features}, outputs: {self.output_features!r})"
            if self.output_features
            else f"DeploymentSchema(inputs: {self.input_features})"
        )
