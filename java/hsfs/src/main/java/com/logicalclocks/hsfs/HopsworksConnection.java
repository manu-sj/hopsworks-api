/*
 *  Copyright (c) 2025. Hopsworks AB
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *  See the License for the specific language governing permissions and limitations under the License.
 *
 */

package com.logicalclocks.hsfs;

import com.logicalclocks.hsfs.metadata.Credentials;
import com.logicalclocks.hsfs.metadata.HopsworksClient;
import com.logicalclocks.hsfs.metadata.HopsworksHttpClient;
import com.logicalclocks.hsfs.metadata.HopsworksInternalClient;
import lombok.Builder;
import software.amazon.awssdk.regions.Region;

import java.io.IOException;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.cert.CertificateException;

public class HopsworksConnection extends HopsworksConnectionBase {

  @Builder
  public HopsworksConnection(String host, int port, String project, Region region, SecretStore secretStore,
      boolean hostnameVerification, String trustStorePath, String certPath, String apiKeyFilePath, String apiKeyValue)
      throws IOException, FeatureStoreException, CertificateException, KeyStoreException, NoSuchAlgorithmException {
    this.host = host;
    this.port = port;
    this.project = getProjectName(project);
    this.region = region;
    this.secretStore = secretStore;
    this.hostnameVerification = hostnameVerification;
    this.trustStorePath = trustStorePath;
    this.certPath = certPath;
    this.apiKeyFilePath = apiKeyFilePath;
    this.apiKeyValue = apiKeyValue;

    HopsworksClient.setupHopsworksClient(host, port, region, secretStore,
        hostnameVerification, trustStorePath, this.apiKeyFilePath, this.apiKeyValue);
    this.projectObj = getProject();
    HopsworksClient.getInstance().setProject(this.projectObj);
    if (!System.getProperties().containsKey(HopsworksInternalClient.REST_ENDPOINT_SYS)) {
      Credentials credentials = HopsworksClient.getInstance().getCredentials();
      HopsworksHttpClient hopsworksHttpClient =  HopsworksClient.getInstance().getHopsworksHttpClient();
      hopsworksHttpClient.setTrustStorePath(credentials.gettStore());
      hopsworksHttpClient.setKeyStorePath(credentials.getkStore());
      hopsworksHttpClient.setCertKey(credentials.getPassword());
      HopsworksClient.getInstance().setHopsworksHttpClient(hopsworksHttpClient);
    }
  }

  /**
   * Retrieve the project feature store.
   *
   * @return FeatureStore object.
   * @throws IOException Generic IO exception.
   * @throws FeatureStoreException If client is not connected to Hopsworks
   */
  public FeatureStore getFeatureStore() throws IOException, FeatureStoreException {
    return getFeatureStore(rewriteFeatureStoreName(project));
  }

  /**
   * Retrieve a feature store based on name. The feature store needs to be shared with
   * the connection's project. The name is the project name of the feature store.
   *
   * @param name the name of the feature store to get the handle for
   * @return FeatureStore object.
   * @throws IOException Generic IO exception.
   * @throws FeatureStoreException If client is not connected to Hopsworks
   */
  public FeatureStore getFeatureStore(String name) throws IOException, FeatureStoreException {
    return featureStoreApi.get(rewriteFeatureStoreName(name), FeatureStore.class);
  }
}
