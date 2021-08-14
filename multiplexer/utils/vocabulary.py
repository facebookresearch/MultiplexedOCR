# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random


class Vocabulary:
    def __init__(self, file_name=None):
        if file_name is not None:
            self.load(file_name)

    def auto_correct(self, word, max_edit_dist, max_ratio=0.35, unique=True):
        upper_word = word.upper()
        # Return if fully matched
        if upper_word in self.vocabulary:
            return word
        # The following cases have edit dist >= 1
        word_len = len(word)
        best_word = upper_word
        best_edit_dist = min(max_edit_dist, word_len * max_ratio)
        count_best = 0

        for voc in self.vocabulary:
            if abs(len(voc) - word_len) > best_edit_dist:
                continue
            current_edit_dist = self.edit_dist(upper_word, voc)
            if current_edit_dist < best_edit_dist:
                best_edit_dist = current_edit_dist
                best_word = voc
                count_best = 1
            elif current_edit_dist == best_edit_dist:
                # Classical uniformly pick min
                count_best += 1
                if random.random() < 1.0 / count_best:
                    best_word = voc

        if unique and count_best > 1:
            return word

        if best_word != upper_word:
            if word.islower():
                best_word = best_word.lower()
            elif word.isupper():
                pass
            elif word[0].isupper() and word[1:].islower():
                best_word = best_word[0] + best_word[1:].lower()
            else:
                count_lower = 0
                count_upper = 0
                for ch in word:
                    if ch.isupper():
                        count_upper += 1
                    else:
                        count_lower += 1
                if count_upper <= count_lower:
                    best_word = best_word.lower()
            print(
                "[Info] Auto-corrected {} to {} with {} choice(s) of edit dist {}".format(
                    word, best_word, count_best, best_edit_dist
                )
            )
            return best_word
        else:
            return word

    def edit_dist(self, word, voc):
        N1 = len(word) + 1
        N2 = len(voc) + 1
        d = [[0 for j in range(N2)] for i in range(N1)]
        for i in range(N1):
            d[i][0] = i
        for j in range(N2):
            d[0][j] = j
        for i in range(1, N1):
            for j in range(1, N2):
                if word[i - 1] == voc[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(d[i][j - 1], d[i - 1][j], d[i - 1][j - 1]) + 1

        return d[N1 - 1][N2 - 1]

    def load(self, file_name):
        with open(file_name, "r") as f:
            self.vocabulary = [voc.rstrip() for voc in f.readlines()]
