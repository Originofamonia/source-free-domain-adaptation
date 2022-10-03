import itertools
import numpy as np
from copy import deepcopy


class Solution:
    def global_alignment(self, seq_a: str, seq_b: str, 
        substitution: dict, gap: int):  # -> list(tuple)
        """
        needleman-wunsch algorithm:
        1. substitution matrix
        2. fill matrix
        3. traceback
        """
        seq_a = seq_a.lower()
        seq_b = seq_b.lower()
        f = np.zeros((len(seq_a) + 1, len(seq_b) + 1))

        # from left: 1, from diag: 2, from upper: 3
        arrow = np.zeros((len(seq_a) + 1, len(seq_b) + 1))

        for i in range(1, f.shape[0]):
            f[i, 0] = i * gap
            arrow[i, 0] = '3'

        for j in range(1, f.shape[1]):
            f[0, j] = j * gap
            arrow[0, j] = '1'

        # fill matrix
        self.needle_fill_matrix2(seq_a, seq_b, sub, gap, f, arrow)
        print(f)

        arrow = np.asarray(arrow, dtype=int)
        print(arrow)

        # traceback
        aligns = []
        i, j = len(seq_a), len(seq_b)
        part_a, part_b = "", ""

        self.dfs(seq_a, seq_b, f, arrow, aligns, i, j, part_a, part_b)  # change from 0
        print(aligns)
        return aligns
    
    def dfs(self, seq_a, seq_b, f, arrow, aligns, i, j, part_a, part_b):

        direc = arrow[i, j]
        # direc = self.peek(f, arrow, i, j)
        print(i, j, direc)

        if len(part_a) == len(part_b) and i == 0 and j == 0:
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

    def needle_fill_matrix(self, seq_a, seq_b, sub, gap, f, arrow):
        """
        arrow points down, diag, right
        """
        for i in range(1, f.shape[0]):
            for j in range(1, f.shape[1]):
                upper = f[i - 1, j] + gap
                left = f[i, j - 1] + gap
                diag = f[i - 1, j - 1] + sub[seq_a[i - 1]][seq_b[j - 1]]
                listy = np.asarray([left, diag, upper])
                f[i, j] = np.max(listy)
                max_indices = (np.squeeze(np.argwhere(listy == np.amax(listy))) + 1).tolist()
                if type(max_indices) == int:
                    max_indices = str(max_indices)
                else:
                    max_indices = ''.join(map(str, max_indices))

                for idx in max_indices:
                    if idx == '1':  # from left
                        arrow[i, j-1] = str(int(arrow[i, j-1])) + '1'
                    elif idx == '2':  # from diag
                        arrow[i-1, j-1] = str(int(arrow[i-1, j-1])) + '2'
                    elif idx == '3':  # from upper
                        arrow[i-1, j] = str(int(arrow[i-1, j])) + '3'
    
    def needle_fill_matrix2(self, seq_a, seq_b, sub, gap, f, arrow):
        """
        arrow points left, diag, up
        """
        for i in range(1, f.shape[0]):
            for j in range(1, f.shape[1]):
                upper = f[i - 1, j] + gap
                left = f[i, j - 1] + gap
                diag = f[i - 1, j - 1] + sub[seq_a[i - 1]][seq_b[j - 1]]
                listy = np.asarray([left, diag, upper])
                f[i, j] = np.max(listy)
                max_indices = (np.squeeze(np.argwhere(listy == np.amax(listy))) + 1).tolist()
                if type(max_indices) == int:
                    max_indices = str(max_indices)
                else:
                    max_indices = ''.join(map(str, max_indices))

                arrow[i, j] = max_indices

    def peek(self, f, arrow, i, j):
        """
        peek left, diag, upper's arrow, return direction
        """
        direc = ''
        left = arrow[i, j - 1]
        diag = arrow[i-1, j-1]
        upper = arrow[i-1, j]
        if '1' in str(left):
            direc += '1'
        if '2' in str(diag):
            direc += '2'
        if '3' in str(upper):
            direc += '3'
        return direc

if __name__ == '__main__':
    sol = Solution()
    seq_a = 'AAAAAA'
    seq_b = 'Ggggg'
    sub = {
        'a': {'a': 1, 't': -1, 'c': -1, 'g': -1},
        't': {'a': -1, 't': 1, 'c': -1, 'g': -1},
        'c': {'a': -1, 't': -1, 'c': 1, 'g': -1},
        'g': {'a': -1, 't': -1, 'c': -1, 'g': 1},
        }
    gap = -2
    sol.global_alignment(seq_a, seq_b, sub, gap)
