import numpy as np

class StabilityAnalyzer:
    def __init__(self):
        pass

    def get_convex_hull_2d(self, points):
        """
        Computes the 2D convex hull for a small number of points (2, 3, or 4).
        Points should be a list of [x, y] coordinates.
        Returns the sorted list of points forming a counterclockwise polygon.
        """
        if len(points) <= 2:
            return points
            
        # Find centroid
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        
        # Sort points by angle relative to centroid to get counterclockwise order
        sorted_points = sorted(points, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
        
        # For 3 or 4 points on a quadruped, they always form a convex shape
        # because the legs are arranged in a rectangle.
        return sorted_points

    def is_point_inside_polygon(self, point, polygon):
        """
        Checks if a 2D point [x, y] is inside a counterclockwise 2D polygon.
        Returns True/False.
        """
        if len(polygon) < 3:
            return False
            
        px, py = point
        n = len(polygon)
        
        # For a counterclockwise convex polygon, the point must be to the left of all edges
        for i in range(n):
            ax, ay = polygon[i]
            bx, by = polygon[(i + 1) % n]
            
            # Cross product (B - A) x (P - A)
            cross_product = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
            
            # If cross product is negative, point is on the right side of the edge (outside)
            if cross_product < -1e-6:
                return False
                
        return True

    def point_to_segment_distance(self, point, seg_a, seg_b):
        """
        Computes the minimum distance from a point to a 2D line segment AB.
        Returns the distance and the closest point on the segment.
        """
        px, py = point
        ax, ay = seg_a
        bx, by = seg_b
        
        ab_x, ab_y = bx - ax, by - ay
        ap_x, ap_y = px - ax, py - ay
        
        ab_lensq = ab_x**2 + ab_y**2
        if ab_lensq < 1e-8:
            return np.sqrt(ap_x**2 + ap_y**2), seg_a
            
        # Projection factor t
        t = (ap_x * ab_x + ap_y * ab_y) / ab_lensq
        t = np.clip(t, 0.0, 1.0)
        
        closest_x = ax + t * ab_x
        closest_y = ay + t * ab_y
        
        dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
        return dist, [closest_x, closest_y]

    def compute_stability(self, com_xy, foot_contacts):
        """
        com_xy: [x, y] coordinates of Center of Mass projected on ground
        foot_contacts: dict of foot positions and contact state:
                       {'FL': ([x, y], in_contact_bool), ...}
                       
        returns: dict containing:
                 - 'support_polygon': list of CCW sorted [x, y] coordinates of contacting feet
                 - 'stability_margin': float (min distance from COM to polygon edge, negative if outside)
                 - 'state': 'stable' | 'warning' | 'unstable'
        """
        # 1. Filter feet currently in contact
        contact_points = []
        for leg, (pos, in_contact) in foot_contacts.items():
            if in_contact:
                contact_points.append(pos[:2]) # keep X-Y only
                
        num_contacts = len(contact_points)
        
        if num_contacts < 3:
            # Cannot form a polygon with fewer than 3 contacts
            return {
                'support_polygon': contact_points,
                'stability_margin': -999.0 if num_contacts == 0 else -0.1,
                'state': 'unstable'
            }
            
        # 2. Get CCW Convex Hull
        polygon = self.get_convex_hull_2d(contact_points)
        
        # 3. Calculate distance to each segment
        min_dist = float('inf')
        closest_pt = None
        
        n = len(polygon)
        for i in range(n):
            dist, pt = self.point_to_segment_distance(com_xy, polygon[i], polygon[(i + 1) % n])
            if dist < min_dist:
                min_dist = dist
                closest_pt = pt
                
        # 4. Check if inside
        inside = self.is_point_inside_polygon(com_xy, polygon)
        
        # Sign the stability margin
        stability_margin = min_dist if inside else -min_dist
        
        # Determine safety state
        # 0.03m (3cm) is a standard safety threshold for small quadrupeds
        if stability_margin > 0.03:
            state = 'stable'
        elif stability_margin > 0.0:
            state = 'warning'
        else:
            state = 'unstable'
            
        return {
            'support_polygon': polygon,
            'stability_margin': float(stability_margin),
            'state': state,
            'closest_point': closest_pt
        }
