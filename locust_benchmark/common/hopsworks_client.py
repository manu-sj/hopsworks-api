import datetime
import random
import string

import numpy as np
import pandas as pd
import os


import hopsworks


class HopsworksClient:
    def __init__(self, environment=None):
        print(
            f'project={os.environ.get("HOPSWORKS_PROJECT", None)}, host={os.environ.get("HOPSWORKS_HOST", None)}, port={os.environ.get("HOPSWORKS_PORT", 443)}, api_key_value={os.environ.get("HOPSWORKS_API_KEY", None)}, engine=python'
        )
        self.project = hopsworks.login(
            project=os.environ.get("HOPSWORKS_PROJECT", None),
            host=os.environ.get("HOPSWORKS_HOST", None),
            port=os.environ.get("HOPSWORKS_PORT", 443),
            api_key_value=os.environ.get("HOPSWORKS_API_KEY", None),
            engine="python",
        )
        self.fs = self.project.get_feature_store()

        # test settings
        self.external = os.environ.get("EXTERNAL", False)
        self.rows = os.environ.get("rows", 1_000_000)
        self.schema_repetitions = os.environ.get("schema_repetitions", 1)
        self.recreate_feature_group = os.environ.get("recreate_feature_group", False)
        self.batch_size = os.environ.get("batch_size", 100)
        self.tablespace = os.environ.get("tablespace", None)

    def get_or_create_fg(self):
        locust_fg = self.fs.get_or_create_feature_group(
            name="locust_fg",
            version=1,
            primary_key=["ip"],
            online_enabled=True,
            stream=True,
            online_config={"table_space": self.tablespace} if self.tablespace else None,
        )
        return locust_fg

    def insert_data(self, locust_fg):
        if locust_fg.id is not None and self.recreate_feature_group:
            locust_fg.delete()
            locust_fg = self.get_or_create_fg()
        if locust_fg.id is None:
            df = self.generate_insert_df(self.rows, self.schema_repetitions)
            locust_fg.insert(df, write_options={"internal_kafka": not self.external})
        return locust_fg

    def get_or_create_fv(self, fg=None):
        if fg is None:
            fg = self.get_or_create_fg()
        return self.fs.get_or_create_feature_view(
            name="locust_fv", version=1, query=fg.select_all()
        )

    def close(self):
        if self.project is not None:
            hopsworks.logout()

    def generate_insert_df(self, rows, schema_repetitions):
        data = {"ip": range(0, rows)}
        df = pd.DataFrame.from_dict(data)

        for i in range(0, schema_repetitions):
            df["rand_ts_1_" + str(i)] = datetime.datetime.now()
            df["rand_ts_2_" + str(i)] = datetime.datetime.now()
            df["rand_int_1" + str(i)] = np.random.randint(0, 100000)
            df["rand_int_2" + str(i)] = np.random.randint(0, 100000)
            df["rand_float_1" + str(i)] = np.random.uniform(low=0.0, high=1.0)
            df["rand_float_2" + str(i)] = np.random.uniform(low=0.0, high=1.0)
            df["rand_string_1" + str(i)] = "".join(
                random.choices(string.ascii_lowercase, k=5)
            )
            df["rand_string_2" + str(i)] = "".join(
                random.choices(string.ascii_lowercase, k=5)
            )
            df["rand_string_3" + str(i)] = "".join(
                random.choices(string.ascii_lowercase, k=5)
            )
            df["rand_string_4" + str(i)] = "".join(
                random.choices(string.ascii_lowercase, k=5)
            )

        return df
