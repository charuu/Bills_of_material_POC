import cv2 as cv2
import numpy as np
import math
from shapely.geometry import LineString,Polygon,Point
import fitz
from rtree import index  # R-tree for fast spatial indexing
import networkx as nx
from shapely.ops import unary_union


def round_coords(line, decimals=6):
    return LineString([(round(x, decimals), round(y, decimals)) for x, y in line.coords])
    
def line_intersects_rect(arrow_boxes,p1,p2,main_image):
    h_main,w_main,_ = main_image.shape
    DPI = 300
    scale = DPI / 72
    (x0, y0) = p1
    (x1, y1) = p2
    sx0, sy0, sx1, sy1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
    side_type = None
    
    for arrow_x,arrow_y,arrow_h,arrow_w,score,angle in arrow_boxes :
        roi_rect_h = 0, arrow_y/scale, w_main/scale, (arrow_y+arrow_h)/scale 
        roi_h = fitz.Rect(*roi_rect_h)
       # cv2.rectangle(main_image, (0,int(arrow_y)), (int(w_main) , int(arrow_y + arrow_w) ), (0, 0, 255), 2)
        
        roi_rect_v = arrow_x/scale, 0, (arrow_x+arrow_w)/scale , (h_main)/scale
        roi_v = fitz.Rect(*roi_rect_v)
       # cv2.rectangle(main_image, (int(arrow_x),0), (int(arrow_x+arrow_h) , int(h_main) ), (0, 0, 255), 2)
    
        orientation_h = (roi_h.contains(fitz.Point(x0, y0)) and roi_h.contains(fitz.Point(x1, y1)))
        orientation_v = (roi_v.contains(fitz.Point(x0, y0)) and roi_v.contains(fitz.Point(x1, y1)))
        #print(orientation_h or orientation_v)
        if orientation_h or orientation_v:
            polygon = Polygon([(arrow_x,arrow_y), (arrow_x+arrow_w, arrow_y), (arrow_x+arrow_w, arrow_y+arrow_h), ( arrow_x, arrow_y+arrow_h),(arrow_x,arrow_y)])
            pts = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.polylines(main_image, [pts], isClosed=True, color=(255, 255, 0), thickness=12)

            min_x, min_y, max_x, max_y = polygon.bounds # Get polygon center
         
            line2 = LineString([(sx0, sy0), (sx1, sy1)])
            cv2.line(main_image, (int(sx0), int(sy0)),  (int(sx1), int(sy1)), color=(0, 255, 255), thickness=12)
            
            buffer_size = 0.01
            intersection = polygon.intersection(line2)
      
            for pt in intersection.coords:
                point = Point(pt)
                if point.y == min_y:
                    side_type ="TOP"
                   # print(f"Point {pt} is on the TOP side of the polygon")
                elif point.y == max_y:
                    side_type ="BOTTOM"
                   # print(f"Point {pt} is on the BOTTOM side of the polygon")
                elif point.x == min_x:
                    side_type ="LEFT"
                   # print(f"Point {pt} is on the LEFT side of the polygon")
                elif point.x == max_x:
                    side_type ="RIGHT"
                  #  print(f"Point {pt} is on the RIGHT side of the polygon")
                else:
                    side_type="INSIDE"
            
            if (intersection or line2.touches(polygon)) :
                #if (side_type=="INSIDE") :
                  #  return (False,None)
            #    else:
                return (True,side_type)
            

    return (False,None)
def rotate_image_with_padding(image, angle):
    """
    Rotates an image and pads it to ensure the full image is retained.
    
    :param image: Input image (NumPy array)
    :param angle: Rotation angle in degrees (positive = counterclockwise, negative = clockwise)
    :return: Rotated and padded image
    """
    (h, w) = image.shape[:2]  # Get original width and height
    
    center = (w // 2, h // 2)  # Find the center of the image

    # Compute the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounding box size
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to shift the image to the center
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
   
    # Rotate and pad the image
    rotated = cv2.warpAffine(image, M, (new_w, new_h),borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255,))  # White padding
    return rotated

def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    x1, y1, h, w, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    x2, y2 = x1 + w, y1 + h

    # Compute area
    areas = w * h
    indices = np.argsort(scores)[::-1]  # Sort by confidence

    picked = []
    while len(indices) > 0:
        i = indices[0]  # Pick the highest-scoring box
        picked.append(boxes[i])

        # Compute IoU
        xx1 = np.maximum(x1[i], x1[indices[1:]])
        yy1 = np.maximum(y1[i], y1[indices[1:]])
        xx2 = np.minimum(x2[i], x2[indices[1:]])
        yy2 = np.minimum(y2[i], y2[indices[1:]])

        w_inter = np.maximum(0, xx2 - xx1)
        h_inter = np.maximum(0, yy2 - yy1)
        intersection = w_inter * h_inter

        iou = intersection / (areas[i] + areas[indices[1:]] - intersection)
        
        # Keep boxes with IoU < threshold
        indices = indices[np.where(iou < overlapThresh)[0] + 1]

    return picked
    
def angle_between_lines(line1, line2):
    # Convert LineString to vectors
    x1, y1, x2, y2 = *line1.coords[0], *line1.coords[1]
    x3, y3, x4, y4 = *line2.coords[0], *line2.coords[1]

    # Compute direction vectors
    v1 = (x2 - x1, y2 - y1)
    v2 = (x4 - x3, y4 - y3)

    # Compute dot product and magnitudes
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    # Prevent division by zero
    if mag_v1 == 0 or mag_v2 == 0:
        return None  # Undefined angle

    # Clamp value to [-1,1] to avoid domain error in acos()
    cos_theta = max(-1, min(1, dot_product / (mag_v1 * mag_v2)))

    # Compute angle in radians and convert to degrees
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)

    return angle_deg
    
def build_graph(lines):
    """
    Build a graph where intersections are nodes and line segments are edges.
    
    Parameters:
    - lines (list of LineString): List of line segments.
    
    Returns:
    - Graph (networkx.Graph): Graph representation of intersections.
    """
    graph = nx.Graph()
    spatial_index = index.Index()

    # Add lines to R-tree for fast lookup
    for i, line in enumerate(lines):
        spatial_index.insert(i, line.bounds)

    # Check intersections efficiently
    for i, line1 in enumerate(lines):
        possible_matches = list(spatial_index.intersection(line1.bounds))

        for j in possible_matches:
            if i >= j:  # Avoid duplicate checks
                continue

            line2 = lines[j]
            intersection = line1.intersection(line2)

            if intersection.geom_type == "Point":
                graph.add_node(tuple(intersection.coords[0]))  # Add intersection point as a node
                graph.add_edge(tuple(line1.coords[0]), tuple(line1.coords[1]))  # Add edges
                graph.add_edge(tuple(line2.coords[0]), tuple(line2.coords[1]))

    return graph

def find_triangles(graph):
    """
    Find all unique triangles in the graph.
    
    Parameters:
    - graph (networkx.Graph): Graph of intersections.
    
    Returns:
    - List of Polygon objects representing triangles.
    """
    triangles = []
    
    for cycle in nx.cycle_basis(graph):  # Finds all cycles in the graph
        if len(cycle) == 3:  # Check if it's a triangle
            triangles.append(Polygon(cycle))
    
    return triangles
    
def sort_coordinates(line):
    """Sort the coordinates of the line in ascending order to avoid reverse direction mismatch."""
    coords = list(line.coords)
    return tuple(sorted(coords))


def remove_triangle_lines(lines, triangles):
    """
    Removes lines that are part of detected triangles.

    Parameters:
    - lines (list of LineString): List of original lines.
    - triangles (list of Polygon): List of detected triangles.

    Returns:
    - List of LineString: Filtered lines that are NOT part of any triangle.
    """
    # Collect all edges that form triangles
    triangle_edges = set()
    for triangle in triangles:
        coords = list(triangle.exterior.coords[:-1])  # Remove duplicate closing point
        for i in range(len(coords)):
            edge = LineString([coords[i], coords[(i+1) % len(coords)]])
            triangle_edges.add(edge)

    normalized_triangle_edges = {sort_coordinates(edge) for edge in triangle_edges}
    
    # Normalize the lines in the list and remove those matching the triangle edges
    remaining_lines = [
        line for line in lines if sort_coordinates(line) not in normalized_triangle_edges
    ]
    
    return remaining_lines
    
    """
    Removes overlapping circles based on IoU threshold.
    
    :param circles: List of Shapely circles
    :param iou_threshold: IoU threshold for suppression
    :return: List of non-overlapping circles
    """
def iou(circle1, circle2):
    """Compute the Intersection over Union (IoU) between two circles."""
    inter_area = circle1.intersection(circle2).area
    union_area = unary_union([circle1, circle2]).area
    return inter_area / union_area if union_area > 0 else 0

def remove_overlapping_circles(circles, iou_threshold=0.3):
    """
    Removes overlapping circles based on IoU threshold.
    
    :param circles: List of Shapely circles
    :param iou_threshold: IoU threshold for suppression
    :return: List of non-overlapping circles
    """
    non_overlapping = []

    # Sort by area (or another criterion)
   # sorted_circles = sorted(circles, key=lambda c: c.area, reverse=True)

    for circle,radius in circles:
        if all(iou(circle, other) < iou_threshold for other in non_overlapping):
            non_overlapping.append([circle,radius])
        else:
            print(f"Removed circle at {circle.centroid} due to overlap.")

    return non_overlapping
