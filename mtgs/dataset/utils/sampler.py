#-------------------------------------------------------------------------------#
# MTGS: Multi-Traversal Gaussian Splatting (https://arxiv.org/abs/2503.12552)   #
# Source code: https://github.com/OpenDriveLab/MTGS                             #
# Copyright (c) OpenDriveLab. All rights reserved.                              #
#-------------------------------------------------------------------------------#
import random
import numpy as np

class CameraSampler:
    
    def __init__(self, dataset, in_order: bool = False):
        self.dataset = dataset
        self.unseen_cameras = [i for i in range(len(self.dataset))]
        self.in_order = in_order

    def get_next_image_idx(self) -> int:

        if self.in_order:
            image_idx = self.unseen_cameras.pop(0)
        else:
            image_idx = self.unseen_cameras.pop(random.randint(0, len(self.unseen_cameras) - 1))
        if len(self.unseen_cameras) == 0:
            self.unseen_cameras = [i for i in range(len(self.dataset))]

        return image_idx

class MultiTraversalBalancedSampler:

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.dataparser_outputs = self.dataset._dataparser_outputs

        self.traversals = set(self.dataparser_outputs.travel_ids)
        
        travel_ids = np.array(self.dataparser_outputs.travel_ids)
        self.traversal_counts = {
            traversal: (travel_ids == traversal).sum() for traversal in self.traversals}
        self.traversal_indices = {
            traversal: (np.where(travel_ids == traversal)[0]).tolist() for traversal in self.traversals
        }

        self.unseen_traversals = list(self.traversals)
        self.unseen_per_traversal_images = {
            traversal: self.traversal_indices[traversal].copy() for traversal in self.traversals
        }

    def get_next_traversal(self):
        traversal = self.unseen_traversals.pop(random.randint(0, len(self.unseen_traversals) - 1))
        if len(self.unseen_traversals) == 0:
            self.unseen_traversals = list(self.traversals)
        return traversal
    
    def get_next_image_idx(self):
        travel_id = self.get_next_traversal()
        image_idx = self.unseen_per_traversal_images[travel_id].pop(random.randint(0, len(self.unseen_per_traversal_images[travel_id]) - 1))
        if len(self.unseen_per_traversal_images[travel_id]) == 0:
            self.unseen_per_traversal_images[travel_id] = self.traversal_indices[travel_id].copy()
        return image_idx
