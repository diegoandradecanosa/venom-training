#
# Copyright (C) 2024
#  Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()

df = pd.read_csv("result/bench.csv")
print(df)
width = 0.5

figure, ax = plt.subplots(1, 1, figsize=(11,8))
bottom = np.zeros(4)

tmp=df[(df.algo==0) & (df.v==64)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
print(tmp)
ax.set_title("BERT-large, bs=16", fontsize=14)
ax_df = tmp.plot.bar(x='m',y='mean', ax=ax, legend=False)
ax_df.set_xticklabels(["dense", "64:2:8", "64:2:16", "64:2:32"], fontsize=14)
x_axis = ax_df.axes.get_xaxis()
x_label = x_axis.get_label()
x_label.set_visible(False)

figure.tight_layout()
ax.set_xlabel("Sparsity", fontsize=14)
ax.set_ylabel("Latency(ms)", fontsize=14)

figure.tight_layout()
plt.savefig('result/bench_v64.pdf')


##############
figure, ax = plt.subplots(1, 1, figsize=(11,8))
bottom = np.zeros(4)

tmp=df[(df.algo==0) & (df.v==128)].sort_values(by="mean", ascending=False).groupby(by=["m"]).first().reset_index()
print(tmp)
ax.set_title("BERT-large, bs=16", fontsize=14)
ax_df = tmp.plot.bar(x='m',y='mean', ax=ax, legend=False)
ax_df.set_xticklabels(["dense", "128:2:8", "128:2:16", "128:2:32"], fontsize=14)
x_axis = ax_df.axes.get_xaxis()
x_label = x_axis.get_label()
x_label.set_visible(False)

#plt.xlabel("Sparsity", fontsize=14)
plt.ylabel("Latency(ms)", fontsize=14)

figure.tight_layout()
plt.savefig('result/bench_v128.pdf')