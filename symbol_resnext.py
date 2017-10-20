# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
import json

if __name__ == "__main__":
    json_network = json.load(open('model/resnext-101-64x4d-symbol.json', 'r'))
    network = mx.symbol.load_json(json.dumps(json_network))
    tmp = mx.viz.plot_network(network, shape={'data': (1, 3, 368, 368)}, save_format='png')
    tmp.view()
