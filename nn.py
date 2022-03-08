import math

def dist(self_x,self_y,n_x,n_y):
    return math.abs(self_x-n_x) + math.abs(self_y-n_y)

if __name__ == "__main__":
    row_num = len(grid)
    col_num = len(grid[0])

    self_x_pos, self_y_pos = r
    for r in range(len(row_num)):
        for c in range(len(col_num)):
            if grid[r][c] == '1':
                self_x_pos = c
                self_y_pos = r

    left = self_x_pos - 1
    right = col_num - self_x_pos

    above = self_y_pos - 1
    below = row_num - self_y_pos

    l = list()
    for above_step in range(above):
        for below_step in range(below):
            for left_step in range(left):
                for right_step in range(right):
                    l = check(grid, below_step, right_step, self_x_pos, self_y_pos,l)
                    l = check(grid, above_step, right_step, self_x_pos, self_y_pos,l)
                    l = check(grid, below_step, left_step, self_x_pos, self_y_pos, l)
                    l = check(grid, above_step, right_step, self_x_pos, self_y_pos,l)

def check(grid,h,v,r,c,l):
    if grid[r-v][c] != '2':
        l.append([r+v,c])
        if grid[r+v][c] != '2':
            l.append([r-v,c])
            if grid[r][c-h] != '2':
                l.append([r,c-h])
                if grid[r][c+h] != '2':
                    l.append([r][c+h])
   return l


    for [x,y] in l:
        new_dist = dist(self_x_pos, self_y_pos, x, y)
        if new_dist < min_dist:
            return new_dist
