#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools

import torch


class LanguageList(object):
    def __init__(self, languages):
        self.languages = languages

    @staticmethod
    def concat(language_lists):
        return LanguageList(list(itertools.chain.from_iterable(language_lists)))

    def transpose(self, method):
        return self

    def crop(self, box, keep_ind=None):
        if keep_ind is not None:
            self.languages = [self.languages[i] for i in keep_ind]
        return self

    def resize(self, size, *args, **kwargs):
        return self

    def set_size(self, size):
        pass

    def rotate(self, angle, r_c, start_h, start_w):
        return self

    def __iter__(self):
        return iter(self.languages)

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            selected_languages = [self.languages[item]]
        else:
            # advanced indexing on a single dimension
            selected_languages = []
            # original_item_for_debug = item.clone()  # DEBUG
            if isinstance(item, torch.Tensor) and item.dtype == torch.uint8:
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                assert i < len(self.languages), "i = {}, len(self.languages) = {}".format(
                    i, len(self.languages)
                )
                selected_languages.append(self.languages[i])
        return LanguageList(selected_languages)

    def __len__(self):
        return len(self.languages)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "languages={}, ".format(self.languages)
        s += ")"
        return s
