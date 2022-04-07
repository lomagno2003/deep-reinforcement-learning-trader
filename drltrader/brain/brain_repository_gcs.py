from google.cloud import storage
import os
from pathlib import Path

from drltrader.brain.brain import Brain
from drltrader.brain.brain_repository import BrainRepository
from drltrader.brain.brain_repository_file import BrainRepositoryFile


class BrainRepositoryGoogleCloudStorage(BrainRepository):
    def __init__(self, gcloud_creds_file: str = None):
        self._bucket_name = 'drltrader-brains'
        self._blob_prefix = 'tests'
        self._local_directory = 'temp'
        self._brain_repository_file = BrainRepositoryFile()

        if gcloud_creds_file is None:
            self._storage_client = storage.Client()
        else:
            self._storage_client = storage.Client.from_service_account_json(gcloud_creds_file)

    def load(self, brain_id: str):
        blob_path = f"{self._blob_prefix}/{brain_id}"
        local_path = f"{self._local_directory}/{brain_id}"

        # TODO: Cleanup if exists
        directory_exists = (Path.cwd() / local_path).exists()
        if not directory_exists:
            os.mkdir(local_path)

        bucket = self._storage_client.get_bucket(self._bucket_name)
        blobs = bucket.list_blobs(prefix=blob_path)
        for blob in blobs:
            filename = blob.name[blob.name.rfind('/') + 1:]
            blob.download_to_filename(f"{local_path}/{filename}")

        return self._brain_repository_file.load(brain_id)

    def save(self, brain_id: str, brain: Brain, override: bool = False):
        # FIXME: This is kind of coupling this repository with the file one
        self._brain_repository_file.save(brain_id, brain, override)

        blob_path = f"{self._blob_prefix}/{brain_id}"
        local_path = f"{self._local_directory}/{brain_id}"

        model_file_path = f"./{local_path}/model.zip"
        brain_configuration_file_path = f"./{local_path}/brain_configuration.pickle"
        model_blob_path = f"{blob_path}/model.zip"
        brain_configuration_blob_path = f"{blob_path}/brain_configuration.pickle"

        bucket = self._storage_client.get_bucket(self._bucket_name)

        model_blob = bucket.blob(model_blob_path)
        brain_configuration_blob = bucket.blob(brain_configuration_blob_path)

        model_blob.upload_from_filename(model_file_path)
        brain_configuration_blob.upload_from_filename(brain_configuration_file_path)
