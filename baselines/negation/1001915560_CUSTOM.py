import itertools
import string
import pandas as pd
import numpy as np
from numpy import unravel_index
from copy import deepcopy

class Solution:
    def local_alignment(self, seq_a: str, seq_b: str, 
        sub: dict, gap: int, alphabets):  # -> list(tuple)
        """
        smith-waterman algorithm
        1. sub matrix
        2. fill F matrix
        3. traceback
        """
        seq_a = seq_a.lower()
        seq_b = seq_b.lower()
        f = np.zeros((len(seq_a) + 1, len(seq_b) + 1))
        arrow = np.zeros((len(seq_a) + 1, len(seq_b) + 1))

        for i in range(1, f.shape[0]):
            f[i, 0] = 0

        for j in range(1, f.shape[1]):
            f[0, j] = 0
        
        self.needle_fill_matrix2(seq_a, seq_b, sub, gap, alphabets, f, arrow)
        print(f)

        arrow = np.asarray(arrow, dtype=int)
        print(arrow)

        aligns = []
        indices = np.argwhere(f == f.max())
        # i, j = len(seq_a), len(seq_b)  # change to start from max location
        part_a, part_b = "", ""
        for ii, jj in indices:

            self.dfs(seq_a, seq_b, f, arrow, aligns, ii, jj, part_a, part_b)
        print(aligns)
        return aligns, f
    
    def dfs(self, seq_a, seq_b, f, arrow, aligns, i, j, part_a, part_b):

        direc = arrow[i, j]
        print(i, j, direc)

        if len(part_a) == len(part_b) and f[i, j] == 0:
            aligns.append((deepcopy(part_a[::-1]), deepcopy(part_b[::-1])))
            return

        for k in str(direc):
            if '1' == k:  # from left to right
                part_a += '-'
                part_b += seq_b[j-1]
                self.dfs(seq_a, seq_b, f, arrow, aligns, i, j-1, part_a, part_b)
                part_a = part_a[:-1]
                part_b = part_b[:-1]

            elif '2' == k:  # from diag
                part_a += seq_a[i-1]
                part_b += seq_b[j-1]
                self.dfs(seq_a, seq_b, f, arrow, aligns, i-1, j-1, part_a, part_b)
                part_a = part_a[:-1]
                part_b = part_b[:-1]

            elif '3' == k:  # from upper to down
                part_a += seq_a[i-1]
                part_b += '-'
                self.dfs(seq_a, seq_b, f, arrow, aligns, i-1, j, part_a, part_b)
                part_a = part_a[:-1]
                part_b = part_b[:-1]
            else:
                return
    
    def needle_fill_matrix2(self, seq_a, seq_b, sub, gap, alphabets, f, arrow):
        """
        arrow points left, diag, up
        """
        for i in range(1, f.shape[0]):
            for j in range(1, f.shape[1]):
                upper = f[i - 1, j] + gap
                left = f[i, j - 1] + gap
                diag = f[i - 1, j - 1] + sub[alphabets.index(seq_a[i - 1]), alphabets.index(seq_b[j - 1])]
                listy = np.asarray([0, left, diag, upper])
                f[i, j] = np.max(listy)
                max_indices = (np.squeeze(np.argwhere(listy == np.amax(listy)))).tolist()
                if type(max_indices) == int:
                    max_indices = str(max_indices)
                else:
                    max_indices = ''.join(map(str, max_indices))

                arrow[i, j] = max_indices


def main():
    name = 'qiyuanan'
    name_set = list(set([char for char in name]))
    alphabets = list(string.ascii_lowercase)
    sub = np.zeros((len(alphabets), len(alphabets)))
    np.fill_diagonal(sub, 2)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            if i == j:
                continue
            if alphabets[i] == alphabets[j]:
                sub[i, j] = 2
            elif alphabets[i] in name_set and alphabets[j] in name_set:
                sub[i, j] = 1
            else:
                sub[i, j] = -1
    
    s = np.append(np.array([alphabets]), sub, axis=0)
    alphabets.insert(0, '_')
    s = np.append(np.expand_dims(np.array(alphabets), axis=1), s, axis=1)
    df = pd.DataFrame(s)
    df.to_csv('1001915560_S.txt',sep='\t', index=True, header=True)
    # np.savetxt('1001915560_S.txt', s, delimiter=',', fmt='%s')
    return sub, alphabets


def question42():
    sol = Solution()
    seq_a = 'qiyuanan'
    seq_b = 'thequickbrownfoxjumpsoverthelazydog'
    gap = -2
    sub, alphabets = main()
    alphabets.remove('_')
    aligns, d = sol.local_alignment(seq_a, seq_b, sub, gap, alphabets)
    seq_b_list = [cha for cha in seq_b]
    seq_b_list.insert(0, ' ')
    d = np.append(np.array([seq_b_list]), d, axis=0)
    seq_a_list = [cha for cha in seq_a]
    seq_a_list.insert(0, ' ')
    seq_a_list.insert(0, ' ')
    d = np.append(np.expand_dims(np.array(seq_a_list), axis=1), d, axis=1)
    df = pd.DataFrame(d)
    df.to_csv('1001915560_D.txt', sep='\t', index=True, header=True)
    # np.savetxt('1001915560_D.txt', d, delimiter=',', fmt='%s')

if __name__ == '__main__':
    # main()
    question42()
