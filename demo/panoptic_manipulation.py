import torch
import sys

def pull_instanceinfo(instance_map, instance_terms, img_info):
    """This is going to take a list of panoptic terms and pull unique counts of each"""
    # print(instance_terms)
    total_pixels = img_info.height * img_info.width
    counts = {}
    counts_area = {}
    for instance in instance_map:
        # print("INSTANCE: {}".format(instance))
        category = instance_terms[instance['category_id']]
        if category not in counts.keys():
            counts[category] = 1
            counts_area[category] = instance['area']
        else:
            counts[category] += 1
            counts_area[category] += instance['area']
    
    counts_area = {key : round(counts_area[key] / total_pixels, 5) for key in counts_area}
    return counts, counts_area
