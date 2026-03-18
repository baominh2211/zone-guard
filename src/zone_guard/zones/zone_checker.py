"""
    Zone Checker - Check whether any Person in the zone
"""

from shapely.geometry import Point, Polygon

class ZoneChecker:
    def __init__(self, polygon: list[list[float]]):
        self._coords = polygon
        self._polygon = Polygon(polygon)

    def is_inside(self, x: float, y: float) -> bool:
        return self._polygon.contains(Point(x, y))
    
    @staticmethod
    def bbox_to_foot_point(bbox: list, frame_w: int, frame_h: int) -> tuple[float, float]:
        """
        Chuyển bbox [x1,y1,x2,y2] pixel → foot-point normalized.
        
        Foot-point = bottom-center = ((x1+x2)/2, y2)
        Normalized = chia cho frame width/height
        """
        x1, y1, x2, y2 = bbox
        fx = (x1 + x2) / 2 / frame_w
        fy = y2 / frame_h
        return (fx, fy)