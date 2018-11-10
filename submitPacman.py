"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 08 Sep 2018
Description :
Input :
Output :
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

import argparse
import os
import pdb 


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nodes', type=str, help='GPU Compute node')
parser.add_argument('-g', '--gpu', type=str, help='GPU index')
parser.add_argument('-p', '--project', type=str, help='Project name')
parser.add_argument('-s', '--script', type=str, help='Python script')
args = parser.parse_args()

nodes = "ugpc0"+args.nodes
gpu = args.gpu 
project = args.project 
script = args.script 

with open("submit.sh", 'w') as f:  
  f.writelines('#!/bin/bash\n')
  f.writelines('#PBS -l walltime=500:00:00\n')
  f.writelines('cd '+project+"\n")
  f.writelines('CUDA_VISIBLE_DEVICES='+gpu+" python "+script)
os.system("qsub -l nodes="+nodes+ " submit.sh")
