#
#   Copyright 2022 Logical Clocks AB
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

import base64
import os
import textwrap
from pathlib import Path

import requests
from hopsworks_common.client import auth, exceptions
from hopsworks_common.client.istio import base as istio


try:
    import jks
except ImportError:
    pass


class Client(istio.Client):
    REQUESTS_VERIFY = "REQUESTS_VERIFY"
    PROJECT_ID = "HOPSWORKS_PROJECT_ID"
    PROJECT_NAME = "HOPSWORKS_PROJECT_NAME"
    HADOOP_USER_NAME = "HADOOP_USER_NAME"
    HDFS_USER = "HDFS_USER"

    DOMAIN_CA_TRUSTSTORE_PEM = "DOMAIN_CA_TRUSTSTORE_PEM"
    MATERIAL_DIRECTORY = "MATERIAL_DIRECTORY"
    T_CERTIFICATE = "t_certificate"
    K_CERTIFICATE = "k_certificate"
    TRUSTSTORE_SUFFIX = "__tstore.jks"
    KEYSTORE_SUFFIX = "__kstore.jks"
    PEM_CA_CHAIN = "ca_chain.pem"
    CERT_KEY_SUFFIX = "__cert.key"
    MATERIAL_PWD = "material_passwd"
    SECRETS_DIR = "SECRETS_DIR"

    def __init__(self, host, port):
        """Initializes a client being run from a job/notebook directly on Hopsworks."""
        self._host = host
        self._port = port
        self._base_url = "http://" + self._host + ":" + str(self._port)

        trust_store_path = self._get_trust_store_path()
        hostname_verification = (
            os.environ[self.REQUESTS_VERIFY]
            if self.REQUESTS_VERIFY in os.environ
            else "true"
        )
        self._project_id = os.environ[self.PROJECT_ID]
        self._project_name = self._project_name()
        self._auth = auth.ApiKeyAuth(self._get_serving_api_key())
        self._verify = self._get_verify(hostname_verification, trust_store_path)
        self._session = requests.session()

        self._connected = True

    def _project_name(self):
        try:
            return os.environ[self.PROJECT_NAME]
        except KeyError:
            pass

        hops_user = self._project_user()
        hops_user_split = hops_user.split(
            "__"
        )  # project users have username project__user
        project = hops_user_split[0]
        return project

    def _project_user(self):
        try:
            hops_user = os.environ[self.HADOOP_USER_NAME]
        except KeyError:
            hops_user = os.environ[self.HDFS_USER]
        return hops_user

    def _get_trust_store_path(self):
        """Convert truststore from jks to pem and return the location"""
        ca_chain_path = Path(self.PEM_CA_CHAIN)
        if not ca_chain_path.exists():
            self._write_ca_chain(ca_chain_path)
        return str(ca_chain_path)

    def _write_ca_chain(self, ca_chain_path):
        """
        Converts JKS trustore file into PEM to be compatible with Python libraries
        """
        keystore_pw = self._cert_key
        keystore_ca_cert = self._convert_jks_to_pem(
            self._get_jks_key_store_path(), keystore_pw
        )
        truststore_ca_cert = self._convert_jks_to_pem(
            self._get_jks_trust_store_path(), keystore_pw
        )

        with ca_chain_path.open("w") as f:
            f.write(keystore_ca_cert + truststore_ca_cert)

    def _convert_jks_to_pem(self, jks_path, keystore_pw):
        """
        Converts a keystore JKS that contains client private key,
         client certificate and CA certificate that was used to
         sign the certificate to PEM format and returns the CA certificate.
        Args:
        :jks_path: path to the JKS file
        :pw: password for decrypting the JKS file
        Returns:
             strings: (ca_cert)
        """
        # load the keystore and decrypt it with password
        ks = jks.KeyStore.load(jks_path, keystore_pw, try_decrypt_keys=True)
        ca_certs = ""

        # Convert CA Certificates into PEM format and append to string
        for _alias, c in ks.certs.items():
            ca_certs = ca_certs + self._bytes_to_pem_str(c.cert, "CERTIFICATE")
        return ca_certs

    def _bytes_to_pem_str(self, der_bytes, pem_type):
        """
        Utility function for creating PEM files

        Args:
            der_bytes: DER encoded bytes
            pem_type: type of PEM, e.g Certificate, Private key, or RSA private key

        Returns:
            PEM String for a DER-encoded certificate or private key
        """
        pem_str = ""
        pem_str = pem_str + "-----BEGIN {}-----".format(pem_type) + "\n"
        pem_str = (
            pem_str
            + "\r\n".join(
                textwrap.wrap(base64.b64encode(der_bytes).decode("ascii"), 64)
            )
            + "\n"
        )
        pem_str = pem_str + "-----END {}-----".format(pem_type) + "\n"
        return pem_str

    def _get_jks_trust_store_path(self):
        """
        Get truststore location

        Returns:
             truststore location
        """
        t_certificate = Path(self.T_CERTIFICATE)
        if t_certificate.exists():
            return str(t_certificate)
        else:
            username = os.environ[self.HADOOP_USER_NAME]
            material_directory = Path(os.environ[self.MATERIAL_DIRECTORY])
            return str(material_directory.joinpath(username + self.TRUSTSTORE_SUFFIX))

    def _get_jks_key_store_path(self):
        """
        Get keystore location

        Returns:
             keystore location
        """
        k_certificate = Path(self.K_CERTIFICATE)
        if k_certificate.exists():
            return str(k_certificate)
        else:
            username = os.environ[self.HADOOP_USER_NAME]
            material_directory = Path(os.environ[self.MATERIAL_DIRECTORY])
            return str(material_directory.joinpath(username + self.KEYSTORE_SUFFIX))

    def _get_cert_pw(self):
        """
        Get keystore password from local container

        Returns:
            Certificate password
        """
        pwd_path = Path(self.MATERIAL_PWD)
        if not pwd_path.exists():
            username = os.environ[self.HADOOP_USER_NAME]
            material_directory = Path(os.environ[self.MATERIAL_DIRECTORY])
            pwd_path = material_directory.joinpath(username + self.CERT_KEY_SUFFIX)

        with pwd_path.open() as f:
            return f.read()

    def _get_serving_api_key(self):
        """Retrieve serving API key from environment variable."""
        if self.SERVING_API_KEY not in os.environ:
            raise exceptions.InternalClientError("Serving API key not found")
        return os.environ[self.SERVING_API_KEY]
