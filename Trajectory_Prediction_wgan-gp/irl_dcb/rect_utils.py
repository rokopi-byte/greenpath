import math


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersection(line1, line2):
    A, B = line1
    C, D = line2
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def line_coeff(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def line_intersection(line1, line2):
    if not intersection(line1, line2):
        return False
    L1 = line_coeff(*line1)
    L2 = line_coeff(*line2)
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def rect_intersection(line, rect):
    p1, p2 = rect
    wd = p2[0] - p1[0]
    ht = p2[1] - p1[1]
    line1 = (p1, (p1[0] + wd, p1[1]))
    line2 = (p1, (p1[0], p1[1] + ht))
    line3 = ((p1[0], p1[1] + ht), p2)
    line4 = ((p1[0] + wd, p1[1]), p2)

    if line_intersection(line, line1):
        return line_intersection(line, line1)
    elif line_intersection(line, line2):
        return line_intersection(line, line2)
    elif line_intersection(line, line3):
        return line_intersection(line, line3)
    elif line_intersection(line, line4):
        return line_intersection(line, line4)
    else:
        return False


def pointInRect(pt, rect):
    x1, y1 = rect[0]
    x2, y2 = rect[1]
    x_p, y_p = pt
    if x1 < x_p < x2:
        if y1 < y_p < y2:
            return True
    return False


def intersect_rect(pt, rects):
    for el in rects:
        if pointInRect(pt, el):
            return True
    return False


def intersect_list_of_rects(line, rects):
    for el in rects:
        if rect_intersection(line, el):
            return rect_intersection(line, el)

    return False


def distanceFromRect(point, rect):
    x1, y1 = rect[0]
    x2, y2 = rect[1]
    x, y = point

    centerX = (x1 + x2) / 2
    centerY = (y1 + y2) / 2

    return math.sqrt((centerX - x)**2 + (centerY - y)**2)


def checkCategory(point, cats, labels=None, mindistance=100.0, max_rect_return=False):
    min_category = {}
    ret_rect = []
    max_value_rect = 0
    for key, value in cats.items():
        for rect in value:
            if pointInRect(point, rect):
                min_category[key] = min_category.get(key, 0.0) + 1.0
                if max_rect_return:
                    ret_rect = rect
                    max_value_rect = 1.0
            elif distanceFromRect(point, rect) < mindistance:
                value = distanceFromRect(point, rect) / mindistance
                min_category[key] = min_category.get(key, 0.0) + value
                if max_rect_return and value > max_value_rect:
                    max_value_rect = value
                    ret_rect = rect
    if max_rect_return:
        return min_category, ret_rect
    return min_category


def number_max(cats):
    if len(cats) > 0:
        return max(cats, key=cats.get)
    else:
        return 0


def label_max(cats, labels):
    return labels[number_max(cats)]