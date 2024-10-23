# -*- coding: UTF-8 -*-
import numpy as np
import json
import tqdm
import os

file_path1 = "zeroshot & gpt-3.5-turbo & data3000.txt"
file_path2 = "assistant_debate & gpt-4o-mini & data3000.txt"

result = []
with open(file_path1, "r") as f:
    for line in f:
        result.append(line.strip())
total = len(result)
correct = 0
for i in range(total):
    if result[i] == "Correct":
        correct += 1
print("length of data: ", total)
print(f"Accuracy of zeroshot: {correct/total}")

result = []
with open(file_path2, "r") as f:
    for line in f:
        result.append(line.strip())
total = len(result)
correct = 0
for i in range(total):
    if result[i] == "Correct":
        correct += 1
print("length of data: ", total)
print(f"Accuracy of single_debate: {correct/total}")
