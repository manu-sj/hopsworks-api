#
#   Copyright 2021 Logical Clocks AB
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

import logging
import os
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

import humps
import joblib
from hopsworks_common import usage
from hopsworks_common.client.exceptions import ModelRegistryException
from hsml.constants import MODEL
from hsml.model import Model


_logger = logging.getLogger(__name__)


class Model(Model):
    """Metadata object representing an sklearn model in the Model Registry."""

    def __init__(
        self,
        id,
        name,
        version=None,
        created=None,
        creator=None,
        environment=None,
        description=None,
        project_name=None,
        metrics=None,
        program=None,
        user_full_name=None,
        model_schema=None,
        training_dataset=None,
        input_example=None,
        model_registry_id=None,
        tags=None,
        href=None,
        feature_view=None,
        training_dataset_version=None,
        model=None,
        **kwargs,
    ):
        super().__init__(
            id,
            name,
            version=version,
            created=created,
            creator=creator,
            environment=environment,
            description=description,
            project_name=project_name,
            metrics=metrics,
            program=program,
            user_full_name=user_full_name,
            model_schema=model_schema,
            training_dataset=training_dataset,
            input_example=input_example,
            framework=MODEL.FRAMEWORK_SKLEARN,
            model_registry_id=model_registry_id,
            feature_view=feature_view,
            training_dataset_version=training_dataset_version,
            model=model,
        )

    def update_from_response_json(self, json_dict):
        json_decamelized = humps.decamelize(json_dict)
        json_decamelized.pop("framework")
        if "type" in json_decamelized:  # backwards compatibility
            _ = json_decamelized.pop("type")
        self.__init__(**json_decamelized)
        return self

    @usage.method_logger
    def save(
        self,
        model_path=None,
        await_registration=480,
        keep_original_files=False,
        upload_configuration: Optional[Dict[str, Any]] = None,
    ):
        """Persist the model including model files and metadata to the model registry.

        If `model_path` is not provided, the model instance set during model creation is saved as a pickle file.

        # Arguments
            model_path: Local or remote (Hopsworks file system) path to the folder where the model files are located, or path to a specific model file.
            await_registration: Awaiting time for the model to be registered in Hopsworks.
            keep_original_files: If the model files are located in hopsfs, whether to move or copy those files into the Models dataset. Default is False (i.e., model files will be moved)
            upload_configuration: When saving a model from outside Hopsworks, the model is uploaded to the model registry using the REST APIs. Each model artifact is divided into
                chunks and each chunk uploaded independently. This parameter can be used to control the upload chunk size, the parallelism and the number of retries.
                `upload_configuration` can contain the following keys:
                * key `chunk_size`: size of each chunk in megabytes. Default 10.
                * key `simultaneous_uploads`: number of chunks to upload in parallel. Default 3.
                * key `max_chunk_retries`: number of times to retry the upload of a chunk in case of failure. Default 1.

        # Returns
            `Model`: The model metadata object.
        """
        remove_temp_files = False
        if not self._model and not model_path:
            raise ModelRegistryException(
                "Please provide either a path to the saved model using the `model_path` argument in the save function or the instance of the model to be saved to the `model` argument in the `create_model` function."
            )
        elif model_path:
            _logger.info(f"Saving sklearn model in path `{model_path}`")
            model_save_path = model_path
        else:
            _logger.info("Attempting to save provided model instance as a pickle file.")
            try:
                model_directory = TemporaryDirectory()
                model_save_path = os.path.join(
                    model_directory.name, self._name + ".pkl"
                )
                joblib.dump(value=self._model, filename=model_save_path)
                remove_temp_files = (
                    True if not keep_original_files else remove_temp_files
                )
            except Exception as e:
                raise ModelRegistryException(
                    "Unable to save the sklearn model as a pickle file. Please make sure that the model provided can be pickled or try saving the model manually."
                ) from e

        super().save(
            model_path=model_save_path,
            await_registration=await_registration,
            keep_original_files=keep_original_files,
            upload_configuration=upload_configuration,
        )

        if self._model and not model_path and not keep_original_files:
            model_directory.cleanup()

    @usage.method_logger
    def load(self, path):
        """Load the instance of the model saved to the model registry using the default load method."""
        if not self._model:
            if not path:
                downloaded_model_directory = self.download()
            else:
                downloaded_model_directory = path
            try:
                model_path = os.path.join(
                    downloaded_model_directory, self._name + ".pkl"
                )
                self._model = joblib.load(model_path)
                if not path:
                    shutil.rmtree(downloaded_model_directory)
            except Exception as e:
                if not path:
                    shutil.rmtree(downloaded_model_directory)
                raise ModelRegistryException(
                    "Unable to load saved model. Please make sure that the model is saved using the default save method or try loading the model manually."
                ) from e
        return self._model
