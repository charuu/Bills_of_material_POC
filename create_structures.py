import cv2
import numpy as np
import math
from shapely.geometry import LineString,Polygon,Point
import fitz
from rtree import index  # R-tree for fast spatial indexing
import networkx as nx
from shapely.ops import unary_union
from utils import *
from BOM import *
from collections import defaultdict
from itertools import chain

from shapely.geometry import box
# Load the main image and template
def find_components_for_BOM(doc,page,main_image,scale):
    
    template = cv2.imread("arrowhead.png")
    
    # Convert to grayscale for better accuracy
    gray_main = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    edges_image = cv2.Canny(gray_main, np.min(gray_main), 255)
    edges_template = cv2.Canny(template, np.min(gray_template), 255)
    
    _, gray_main_binary = cv2.threshold(gray_main, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, gray_template_binary = cv2.threshold(gray_template, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    h_main,w_main = gray_main.shape
    h,w = gray_template.shape
    
    results = {}  
    center = (w // 2, h // 2)  # Rotation center
    arrows=[]
    
    for angle in range(0, 360, 90):  # Rotate every 30 degrees
        # Get rotation matrix
        rotated = rotate_image_with_padding(gray_template_binary, angle) 
        results= cv2.matchTemplate(gray_main_binary, rotated, cv2.TM_CCOEFF_NORMED)
        h,w = rotated.shape
    
        # Define a threshold for good matches
        threshold = 0.90  # Adjust based on accuracy needs
        locations = np.where(results >= threshold)
    
        for pt in zip(*locations[::-1]):
            score = results[pt[1], pt[0]]
            arrows.append([pt[0],pt[1], h, w,score,angle])
            
    filtered = non_max_suppression(arrows)
    print("Filtered : ",len(filtered))
    
    for i, (arr_x, arr_y, arr_h, arr_w, score, flip_type) in enumerate(filtered):
        cv2.putText(main_image, f"{flip_type}", (int(arr_x), int(arr_y) - 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(main_image, (int(arr_x),int(arr_y)),(int(arr_x + arr_w) , int(arr_y + arr_h)) , (0, 0, 255), 2)
     
    dimension_lines = []
    filtered_lines = []
    triangles = []
    
    drawings = page.get_drawings()
    
    line_items = [item for drawing in drawings for item in drawing["items"] if item[0] == "l"]
    
    for i, line in enumerate(line_items):
        x0, y0, x1, y1 = line[1][0], line[1][1], line[2][0], line[2][1]  # Extract coordinates
        
        intersects_bool, side = line_intersects_rect(filtered, (x0, y0), (x1, y1), main_image)
        sx0, sy0, sx1, sy1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
        
        if intersects_bool:
            filtered_lines.append(LineString([(sx0, sy0), (sx1, sy1)]))
            
            graph = build_graph(filtered_lines)
            dimension_arrow_triangles = find_triangles(graph)
            
            for triangle in dimension_arrow_triangles:
                pts = np.array(triangle.exterior.coords[:-1], dtype=np.int32)
                cv2.polylines(main_image, [pts], isClosed=True, color=(0, 0, 255), thickness=22)
    
            dimension_lines = remove_triangle_lines(filtered_lines, dimension_arrow_triangles)
            
            for line in dimension_lines:
                # Convert LineString to a list of coordinates
                start = tuple(map(int, line.coords[0]))  # First coordinate
                end = tuple(map(int, line.coords[-1]))   # Last coordinate
                
                # Draw the line on the image using cv2.line
                cv2.line(main_image, start, end, color=(255, 0, 0), thickness=12)  # Red line
        
    filtered_line_items = [line for line in line_items if LineString([(line[1][0]* scale, line[1][1]* scale), (line[2][0]* scale, line[2][1]* scale)]) not in filtered_lines]    
    
    extension_lines=[]
    for line in filtered_line_items:
        x0, y0, x1, y1 = line[1][0], line[1][1], line[2][0], line[2][1] 
        sx0, sy0, sx1, sy1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
        
        for tri in dimension_arrow_triangles:
            if tri.intersects(LineString([(sx0, sy0), (sx1, sy1)])):
                extension_lines.append(LineString([(sx0, sy0), (sx1, sy1)]))
                cv2.line(main_image, (int(sx0), int(sy0)), (int(sx1), int(sy1)), (0, 255, 0), 12)
    
    find_part_line_items =[line for line in filtered_line_items if LineString([(line[1][0]* scale, line[1][1]* scale), (line[2][0]* scale, line[2][1]* scale)]) not in extension_lines]    
    
    circles = cv2.HoughCircles(
        gray_main, 
        cv2.HOUGH_GRADIENT, dp=1, minDist=10, 
        param1=250, param2=80, minRadius=40, maxRadius=100
    )
    
    part_item=[]
    part_lines=[]
    part_items =[]
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Convert to integers
        for circle in circles[0, :]:
            x_center, y_center, radius = circle
            x, y, w, h = x_center - radius, y_center - radius, 2 * radius, 2 * radius
            text = page.get_text("text",clip=(int(x/scale),int(y/scale),int((x+w)/scale),int((y+h)/scale)))
            
            if text !="":
                circle_item = Point(x_center, y_center).buffer(radius)
                part_item.append([circle_item,radius,text])
                part_items.append(circle_item)
    
    print(len(part_item))
    #filtered_circles = remove_overlapping_circles(part_item,0.1)
    #print(len(filtered_circles))
    
    for line in find_part_line_items:
        x0, y0, x1, y1 = line[1][0], line[1][1], line[2][0], line[2][1] 
        sx0, sy0, sx1, sy1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
        
        for part,radius,text in part_item:    
            cv2.circle(main_image, (int(part.centroid.x), int(part.centroid.y)), int(radius), (255, 0, 255), 12) 
            if ( part.intersects(Point(sx0, sy0).buffer(5)) ^  part.intersects(Point(sx1, sy1).buffer(5))  ) or (part.contains(Point(sx0, sy0).buffer(5)) ^  part.contains(Point(sx1, sy1).buffer(5))):   
                cv2.line(main_image, (int(sx0), int(sy0)), (int(sx1), int(sy1)), (0, 255, 0), 12)
                part_lines.append(LineString([(sx0, sy0), (sx1, sy1)]))
              
    find_part_arrow_line_items =[line for line in find_part_line_items if LineString([(line[1][0]* scale, line[1][1]* scale), (line[2][0]* scale, line[2][1]* scale)]) not in part_lines]    
    
    arrow_lines=[]
    for line in find_part_arrow_line_items:
        x0, y0, x1, y1 = line[1][0], line[1][1], line[2][0], line[2][1] 
        sx0, sy0, sx1, sy1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale
        for arrow_line in part_lines:
            if (LineString([(sx0, sy0), (sx1, sy1)]).intersects(arrow_line.buffer(10))):
                arrow_lines.append(LineString([(sx0, sy0), (sx1, sy1)]))
    
    graph = build_graph(arrow_lines)
    part_arrow_triangles = find_triangles(graph)
    
    for triangle in part_arrow_triangles:
        pts = np.array(triangle.exterior.coords[:-1], dtype=np.int32)
        cv2.polylines(main_image, [pts], isClosed=True, color=(0, 0, 255), thickness=22)
    
    connecting_part_lines = remove_triangle_lines(part_lines, part_arrow_triangles)
    
    cv2.imwrite("draw_circles_arrowheads_BAE2.png",main_image)

    return dimension_lines,dimension_arrow_triangles,extension_lines,connecting_part_lines,part_arrow_triangles,part_items

def find_text(line_segments, key, text_blocks):
    DPI = 300
    scale = DPI / 72
    
    min_x = min( line.bounds[0] for line in line_segments)
    min_y = min( line.bounds[1] for line in line_segments)
    max_x = max( line.bounds[2] for line in line_segments)
    max_y = max( line.bounds[3] for line in line_segments)
    
    comb_line = LineString([(min_x, min_y), (max_x, max_y)])
    
    block1 = box(min_x, min_y, max_x, max_y)
    centroid_x= (min_x + max_x) / 2
    centroid_y= (min_y + max_y) / 2
    p_l = Point(centroid_x,centroid_y)
    
    intersecting_blocks = []
    find_text=[]
    min_distance = np.iinfo(np.int32).max

    if key in ["pattern1","pattern4"]:
        for i, block in enumerate(text_blocks):
            x, y, x0, y0, text, block_no, block_type = block[:7]
            block2 = box(x*scale, y*scale, x0*scale, y0*scale)
            
            a,b,c,d =line_segments[0].bounds
            l,m,n,o =line_segments[1].bounds
            comb_line2 = LineString([(c,d), (l,m)])
            if comb_line2.intersects(block2) :
                intersecting_blocks.append(block2)
                find_text.append(text.replace('\x01#\x02','tolerance:'))
            
    if key in ["pattern2","pattern3"]:
        for i, block in enumerate(text_blocks):
            x, y, x0, y0, text, block_no, block_type = block[:7]
            block2 = box(x*scale, y*scale, x0*scale, y0*scale)
            block1 = box(min_x-500, min_y, max_x+500, max_y)
            
            if block1.intersects(block2):
                intersecting_blocks.append(block2)
                find_text.append(text.replace('\x01#\x02','tolerance:'))
                
    if key in ["pattern5","pattern6"]:
        for i, block in enumerate(text_blocks):
            x, y, x0, y0, text, block_no, block_type = block[:7]
            block2 = box(x*scale, y*scale, x0*scale, y0*scale)
            block1 = box(min_x, min_y-500, max_x, max_y+500)
            
            if block1.intersects(block2):
                intersecting_blocks.append(block2)
                find_text.append(text.replace('\x01#\x02','tolerance:'))
        
    return find_text

def find_patterns(groups, dimension_lines, dimension_arrow_triangles, dimensions_dict_xy, pattern_map,page):
    read_pattern= defaultdict(list)
    

    filtered = {k:sorted(list(set(v))) for k, v in groups.items() if len(list(set(v))) > 1}

    text_blocks = page.get_text("blocks")
    
    for j, (key, value) in enumerate(filtered.items()):
        read_pattern.clear()
 
        for i in range(len(value)-1):
            dimensions_dict= defaultdict(list)
            
            y1=value[i][1] 
            y2=value[i+1][1] 
            x1=value[i][0]
            x2=value[i+1][0]
    
            line1 = LineString([(value[i]), (value[i+1])])
            line2 = LineString([(value[i+1]), (value[i])])
            
            if line1 in dimension_lines or line2 in dimension_lines:
                key = x1 if y1 == y2 else y1
                read_pattern[key].append("dimension_line")
                 
                for intersecting_dimension_arrows in dimension_arrow_triangles:
                    if line1.intersects(intersecting_dimension_arrows):
                        if y1 == y2:
                            if intersecting_dimension_arrows.centroid.x>=value[i][0] or intersecting_dimension_arrows.centroid.x>=value[i+1][0]:
                                read_pattern[intersecting_dimension_arrows.centroid.x].append("right_arrow")
                            else:
                                read_pattern[intersecting_dimension_arrows.centroid.x].append("left_arrow")
                        else:
                            if intersecting_dimension_arrows.centroid.y>=value[i][1] or intersecting_dimension_arrows.centroid.y>=value[i+1][1]:
                                read_pattern[intersecting_dimension_arrows.centroid.y].append("bottom_arrow")
                            else:
                                read_pattern[intersecting_dimension_arrows.centroid.y].append("top_arrow")
                        
                    
            sorted_values = [read_pattern[k] for k in sorted(read_pattern)]
            flat_list = list(chain(*sorted_values))
            
            for pattern_key, config in pattern_map.items():
                if flat_list in config["patterns"]:
                    dimensions_dict["pattern"].append(pattern_key)
                    line_segments = []
                    for start_offset, end_offset in config["offsets"]:
                        line_segments.append(LineString([value[i + start_offset], value[i + end_offset]]))
            
                    dimensions_dict["coords"].append(line_segments)
                    dimensions_dict["value"].append(find_text(line_segments,pattern_key,text_blocks))
                    dimensions_dict_xy.append(dimensions_dict)
                    read_pattern.clear()
                    break 
       
    return dimensions_dict_xy