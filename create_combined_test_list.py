#!/usr/bin/env python3

import random


with open('images/list/intersection_list.txt', 'r') as f:
    intersection_lines = [line.strip() for line in f if line.strip()]


with open('images/list/training_list.txt', 'r') as f:
    training_lines = [line.strip() for line in f if line.strip()]

print(f'Intersection list (current test): {len(intersection_lines)} images')
print(f'Training list: {len(training_lines)} images')


random.seed(42)  # For reproducibility
sample_size = min(2000, len(training_lines))
sampled_training = random.sample(training_lines, sample_size)


combined_list = intersection_lines + sampled_training


with open('images/list/combined_test_list.txt', 'w') as f:
    for line in combined_list:
        f.write(line + '\n')

print(f'\nCreated combined_test_list.txt with {len(combined_list)} images')
print(f'  - {len(intersection_lines)} from intersection_list.txt')
print(f'  - {len(sampled_training)} from training_list.txt')

