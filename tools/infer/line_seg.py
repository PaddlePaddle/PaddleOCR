
import math


def get_line_para(x1, y1, x2, y2):
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    return A, B, C


def line_length(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_cross_point(a1, b1, c1, a2, b2, c2):
    x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
    y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
    return round(x), round(y)


class LineSeg:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.A, self.B, self.C = get_line_para(x1, y1, x2, y2)
        self.length = line_length(x1, y1, x2, y2)

    def get_cross_point(self, line):
        a1, b1, c1 = self.A, self.B, self.C
        a2, b2, c2 = line.A, line.B, line.C
        x = (b1 * c2 - b2 * c1) / (a1 * b2 - a2 * b1)
        y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
        return round(x), round(y)

    def get_line_vertical(self, pt):
        px, py = pt[0], pt[1]
        AA, BB, CC = 0, 0, 0
        if self.A == 0:
            BB = 0
            AA = -1 * self.B
            CC = self.B * px
        elif self.B == 0:
            AA = 0
            BB = -1 * self.A
            CC = self.A * py
        else:
            AA, BB = -1 * self.B, self.A
            CC = -1 * (AA * px + BB * py)
        return AA, BB, CC

    def left(self):
        if self.x1 > self.x2:
            return self.x2, self.y2
        return self.x1, self.y1

    def right(self):
        if self.x1 > self.x2:
            return self.x1, self.y1
        return self.x2, self.y2

    def top(self):
        if self.y1 > self.y2:
            return self.x2, self.y2
        return self.x1, self.y1

    def bottom(self):
        if self.y1 > self.y2:
            return self.x1, self.y1
        return self.x2, self.y2



