"""
Configuration files for the MCI project

Choose your configuration by placing a " your_platform_name.platform.config" file in the same directory
for example, to choose windows-Y450 configuration, create a file ./windows_Y450.platform.config

@author: Ming Lin
@contact: linmin@umich.edu
"""

import sys
import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../YelabMLib/')
# print("appending path: " +  os.path.dirname(os.path.abspath(__file__)) + '/../../YelabMLib/' +'\n')


import glob
import distutils.dir_util
this_script_path = os.path.dirname(os.path.abspath(__file__))
platform = glob.glob(this_script_path + "/" + "*.platform.config")[0]
platform = os.path.split(platform)[1]

config = {}

config["MingYeWinDesk.platform.config"] = {}
config["YeGPUServer.platform.config"] = {}


config["MingYeWinDesk.platform.config"]["data_repo_dir"] = "d:/tmp/DeepLearningDir/dataset_repo/"
config["YeGPUServer.platform.config"]["data_repo_dir"] = "/data/yelab/dataset_repo/"
data_repo_dir = config[platform]["data_repo_dir"]
# distutils.dir_util.mkpath(data_repo_dir)

config["MingYeWinDesk.platform.config"]["model_repo_dir"] = "d:/tmp/DeepLearningDir/model_repo/"
config["YeGPUServer.platform.config"]["model_repo_dir"] = "/data/minglin/model_repo/"
model_repo_dir = config[platform]["model_repo_dir"]
# distutils.dir_util.mkpath(model_repo_dir)

config["MingYeWinDesk.platform.config"]["output_dir"] = "d:/tmp/DeepLearningDir/output/"
config["YeGPUServer.platform.config"]["output_dir"] = "/data/minglin/output/"
output_dir = config[platform]["output_dir"]
distutils.dir_util.mkpath(output_dir)