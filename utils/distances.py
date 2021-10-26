import math

from haversine import haversine


def slender_distance(p1, p2, center, alpha_1=1, alpha_2=0):
    ang_d = math.radians(get_angle(p1, p2, center))
    radial_d = haversine(p1, p2)
    return alpha_1*ang_d+alpha_2*radial_d


def get_angle(a, b, origin):
    ang = math.degrees(math.atan2(
        b[1]-origin[1], b[0]-origin[0]) - math.atan2(a[1]-origin[1], a[0]-origin[0]))
    ang = abs(ang) if abs(ang) < 180 else 360-abs(ang)
    return ang
