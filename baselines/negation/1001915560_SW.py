import itertools
import numpy as np
from numpy import unravel_index
from copy import deepcopy

class Solution:
    def local_alignment(self, seq_a: str, seq_b: str, 
        sub: dict, gap: int):  # -> list(tuple)
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
        
        self.needle_fill_matrix2(seq_a, seq_b, sub, gap, f, arrow)
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
        return aligns
    
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
    
    def needle_fill_matrix2(self, seq_a, seq_b, sub, gap, f, arrow):
        """
        arrow points left, diag, up
        """
        for i in range(1, f.shape[0]):
            for j in range(1, f.shape[1]):
                upper = f[i - 1, j] + gap
                left = f[i, j - 1] + gap
                diag = f[i - 1, j - 1] + sub[seq_a[i - 1]][seq_b[j - 1]]
                listy = np.asarray([0, left, diag, upper])
                f[i, j] = np.max(listy)
                max_indices = (np.squeeze(np.argwhere(listy == np.amax(listy)))).tolist()
                if type(max_indices) == int:
                    max_indices = str(max_indices)
                else:
                    max_indices = ''.join(map(str, max_indices))

                arrow[i, j] = max_indices


if __name__ == '__main__':
    sol = Solution()
    seq_a = 'cgtgaattcat'
    seq_b = 'gacttac'
    match = 5
    mismatch = -3
    gap = -4
    sub = {
        'a': {'a': match, 't': mismatch, 'c': mismatch, 'g': mismatch},
        't': {'a': mismatch, 't': match, 'c': mismatch, 'g': mismatch},
        'c': {'a': mismatch, 't': mismatch, 'c': match, 'g': mismatch},
        'g': {'a': mismatch, 't': mismatch, 'c': mismatch, 'g': match},
        }
    sol.local_alignment(seq_a, seq_b, sub, gap)