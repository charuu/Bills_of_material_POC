import cv2
import re
import numpy as np
from shapely.geometry import LineString,Polygon,Point,box
import fitz

class DimensionLine:
    def __init__(self, coords, value,orientation, pattern_enum):
        self.coords = coords  # Line coordinates
        self.value = value  # From table
        self.orientation = orientation
        self.pattern = pattern_enum

    def display_info(self):
       # print(f"Coords: {self.coords},\n   Dimension: '{self.dimension}', Tolerance: '{self.tolerance}'\n   Pattern: {self.pattern}")
        print(f"Coords:", self.coords,f"\n   Value:", self.value)
        
    def set_value(self, df, group, value):
        arr = value[0].split('\n')
        if df is not None:
            if 'DIMENSION' in df.columns:
                for a in arr :
                    row = df[df['DIMENSION'] == a]
                    if not row.empty:
                        dim_value = row.iloc[0][group] if not row.empty else None
                        self.value = dim_value
            else:
                self.value=value
class Part:
    def __init__(self, coords, line_ref, quantity):
        self.coords = coords
        self.line = line_ref
        self.quantity = quantity
        
    def __eq__(self, other):
        if isinstance(other, Part):
            return (self.coords == other.coords and
                    self.line == other.line and
                    self.quantity == other.quantity)
        return False
        
    def display_info(self):
        print(f"Coords: {self.coords},\n   Line Ref: '{self.line}',\n   Quantity: {self.quantity}")


class BOMNode:
    def __init__(self, parent=None, bbox=None, dimension_lines=None, part_numbers=None, pattern_enum=None):
        self.parent = parent
        self.neighbour = None
        self.bbox = bbox
        self.dimension_lines= dimension_lines
        self.part_numbers= part_numbers
        self.pattern = pattern_enum
        self.children = []

    def add_child(self, child_node):
        self.children.extend([child_node])

    def __eq__(self, other):
        if isinstance(other, BOMNode):
            return (self.parent == other.parent and
                    self.bbox == other.bbox and
                    self.dimension_lines == other.dimension_lines and self.part_numbers == other.part_numbers)
        return False
        
    def display_info(self, indent=0):
        prefix = " " * indent
        print(f"{prefix}BOMNode:")
        print(f"{prefix}  BBox: {self.bbox}")
        print(f"{prefix}  Parent: {self.parent}")
        if self.neighbour:
            print(f"{prefix}  Neighbour: {self.neighbour}")
        print(f"{prefix}  Dimension Lines:")
        print(f"{prefix}    ", end="")
        self.dimension_lines.display_info()
        print(f"{prefix}  Part Numbers:")
        for part in self.part_numbers:
            print(f"{prefix}    ", end="")
            part.display_info()
       # print(f"{prefix}  Children:")
        #for child in self.children:
        #    print(f"{prefix}    ", end="")
        #    child.display_info()


def is_dimension_line_in_bounding_box(coords, drawing_bbox):
    x_min, y_min, x_max, y_max = drawing_bbox

    for line in coords:
        px_min, py_min, px_max, py_max = line.bounds

        # Check if the entire line is inside the bbox
        if not (x_min <= px_min <= x_max and x_min <= px_max <= x_max and
                y_min <= py_min <= y_max and y_min <= py_max <= y_max):
            return False
    return True
    
def is_dim_horizontal_or_vertical(dim_coords):
    x_min = min([line.bounds[0] for line in dim_coords])
    y_min = min([line.bounds[1] for line in dim_coords])
    x_max = max([line.bounds[2] for line in dim_coords])
    y_max = max([line.bounds[3] for line in dim_coords])

    if x_min == x_max:
        return "vertical"
    elif y_min == y_max:
        return "horizontal"
    
def is_line_parallel_to_dim_line(tri_polygon, bbox, dim_coords, line_items,graph_img):
    dim_orientation = is_dim_horizontal_or_vertical(dim_coords)
    inter_orientation = ""
    DPI = 300
    scale = DPI / 72
    for i, line in enumerate(line_items):
        x0, y0, x1, y1 = line[1][0], line[1][1], line[2][0], line[2][1]  # Extract coordinates
        if tri_polygon.intersects(LineString([(x0*scale,y0*scale),(x1*scale,y1*scale)]).buffer(2)):
            if x0 == x1 :
                inter_orientation = "vertical"
                break
            elif y0 == y1:
                inter_orientation = "horizontal"
                break
                        
    if inter_orientation == dim_orientation:
        cv2.rectangle(graph_img, (int(x0*scale),int(y0*scale)), (int(x1*scale) , int(y1*scale)), (0, 255, 0), 12)      
        return True
    else:
        return False
                
    
def construct_bom_nodes(dimensions_dict, part_groups, part_arrows, drawing_bbox, graph_img, page,df,group):
    bom_nodes = []
    all_parts_in_drawing = []
    drawing_paths = page.get_drawings()
    line_items = [item for drawing in drawing_paths for item in drawing["items"] if item[0] == "l"]
    
    for dim_info in dimensions_dict:
        
        coords = dim_info["coords"][0]
        value = dim_info["value"][0]
        pattern = dim_info["pattern"][0]
        
        if is_dimension_line_in_bounding_box(coords, drawing_bbox):
            bbox = get_bounding_box(coords, pattern, drawing_bbox)
            
            x, y, x2, y2 = bbox
            cv2.rectangle(graph_img, (int(x),int(y)), (int(x2) , int(y2)), (0, 0, 255), 12)
      
            parts = []
            for pg in part_groups:
                tri_polygon = part_arrows[pg.coords.x, pg.coords.y]
                if is_point_in_bbox(pg.coords, bbox) and is_line_parallel_to_dim_line(tri_polygon[0], bbox, coords, line_items, graph_img):
                    parts.append(pg)
                    all_parts_in_drawing.append(pg)
                  
            dim_orientation = is_dim_horizontal_or_vertical(coords)
           # print(value)
            dimension = DimensionLine(coords, value, dim_orientation, pattern)
            dimension.set_value(df,group,value)
            #print(dimension.value)
            node = BOMNode(None,bbox, dimension, parts, pattern)
           # print(node.display_info())
            bom_nodes.append(node)

    distance = []
    diff = [item for item in part_groups if item not in all_parts_in_drawing]
    diff_nodes_list=[]
 
    if diff !=[]:
        for d in diff:
            distance = []
            for node in bom_nodes:
                if (node.dimension_lines.value) :
                    x, y = d.coords.x, d.coords.y
                    dist = Point(x, y).distance(box(node.bbox[0],node.bbox[1],node.bbox[2],node.bbox[3]).centroid)
                    distance.append([dist,  node])
                    
                    
            if distance!=[]:
                min_item = min(distance, key=lambda x: x[0])
                min_dist,  min_node = min_item 
                
                diff_node = BOMNode(None,"",DimensionLine("", None, "", ""), [d], "")
                diff_node.neighbour= min_node
                diff_nodes_list.append(diff_node)
          
    return bom_nodes,diff_nodes_list

def get_bounding_box(coords_list, pattern_enum, drawing_bbox):
    x, y, x2, y2 = drawing_bbox
  #  print(coords_list)
    x_min = min([line.bounds[0] for line in coords_list])
    y_min = min([line.bounds[1] for line in coords_list])
    x_max = max([line.bounds[2] for line in coords_list])
    y_max = max([line.bounds[3] for line in coords_list])
    
    if pattern_enum in ['pattern1','pattern2', 'pattern3']:  # horizontal
        return (x_min , y, x_max, y2)
    elif pattern_enum in ['pattern4','pattern5','pattern6']:  # vertical
        return (x, y_min, x2, y_max)
    else:
        return ("")

def is_point_in_bbox(point, bbox):
    x, y = point.x, point.y
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2
    
def is_bbox_inside(inner, outer):
    if inner !="" and outer!="":
        return (inner[0] >= outer[0] and
                inner[1] >= outer[1] and
                inner[2] <= outer[2] and
                inner[3] <= outer[3])
    else: 
        return False
        
def compare_area(bbox1,bbox2):
    if bbox1.bbox and bbox2.bbox:
        area1 = (bbox1.bbox[2] - bbox1.bbox[0] )*(bbox1.bbox[3] - bbox1.bbox[1])
        area2 = (bbox2.bbox[2] - bbox2.bbox[0] )*(bbox2.bbox[3] - bbox2.bbox[1])
        return area1 < area2

def is_box_intersecting(bbox1,bbox2):
    if bbox1.bbox and bbox2.bbox:
        box1 = box(bbox1.bbox[0] , bbox1.bbox[1] , bbox1.bbox[2]-bbox1.bbox[0] , bbox1.bbox[3]-bbox1.bbox[1])
        box2 =  box(bbox2.bbox[0] , bbox2.bbox[1] , bbox2.bbox[2]-bbox2.bbox[0] , bbox2.bbox[3]-bbox2.bbox[1])
        return box1.intersects(box2)
    
def build_bom_tree(bom_nodes):
    roots = []

    for i, parent in enumerate(bom_nodes):
        for j, child in enumerate(bom_nodes):
            if i == j:
                continue
            if (is_bbox_inside(child.bbox, parent.bbox)) and   (child.dimension_lines.orientation==parent.dimension_lines.orientation):
                
                if (child.parent is None) or (child.parent is not None and compare_area(child.parent, child)):
                    parent.add_child(child)
                    child.parent = parent
                parent.part_numbers = [item for item in parent.part_numbers if item not in child.part_numbers]
                
        if parent.children:
            largest_child = max(parent.children, key=lambda c: (c.bbox[2] - c.bbox[0] )*(c.bbox[3] - c.bbox[1]))
            parent.children = [largest_child]
    
    # Collect nodes that are not children of any other node
    for node in bom_nodes:
        if (node.parent is None) or (node.parent.dimension_lines.value is None):
            if node.dimension_lines is not None:
                roots.append(node)

    return roots

def print_bom_tree(node, level=0):
    indent = "    " * level
    print("\n")
    if (node.parent is not None) :
        print(f"{indent}  Parent: {node.parent.dimension_lines.value}")
    if node.dimension_lines:
        print(f"{indent}  Dimensions:")
        print(f"{indent}    - Value: {node.dimension_lines.value}")
    if node.neighbour:
        print(f"{indent}  Neighbour:")
        print(f"{indent}    - Value: {node.neighbour.dimension_lines.value}")
        
    if node.part_numbers:
        print(f"{indent}  Parts:")
        for part in node.part_numbers:
            print(f"{indent}    - Line: {part.line}, Qty: {part.quantity}")

    for child in node.children:
        print_bom_tree(child, level + 1)

