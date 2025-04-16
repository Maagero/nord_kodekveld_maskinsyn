import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
import shutil
import random
import argparse
from pathlib import Path

# Class mapping according to rdd.yaml
class_mapping = {
    'D00': 0,  # Longitudinal Crack
    'D10': 1,  # Transverse Crack
    'D20': 2,  # Alligator Crack
    'D40': 3,  # Potholes
}

def parse_voc_annotation(xml_file):
    """Parse PascalVOC XML annotation file and extract bounding boxes."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image dimensions
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    
    boxes = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_mapping:
            continue  # Skip classes not in our mapping
            
        class_id = class_mapping[class_name]
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # Convert to YOLOv8 format (normalized center x, center y, width, height)
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        boxes.append((class_id, x_center, y_center, width, height))
    
    return boxes

def convert_to_yolo(voc_dir, model_dir, val_split=0.2):
    """
    Convert PascalVOC dataset to YOLOv8 format.
    
    Args:
        voc_dir: Path to the voc directory containing subdirectories with VOC data
        model_dir: Path to output model directory where YOLO formatted data will be saved
        val_split: Percentage of data to use for validation (0.2 = 20%)
    """
    print(f"Looking for VOC datasets in: {voc_dir}")
    
    # Get all subdirectories in the voc directory
    subdirs = [d for d in os.listdir(voc_dir) if os.path.isdir(os.path.join(voc_dir, d))]
    
    if not subdirs:
        print("No subdirectories found in the voc/ directory.")
        return
    
    for subdir in subdirs:
        print(f"Processing dataset: {subdir}")
        
        # Create output directories
        train_dir = os.path.join(model_dir, subdir, 'train')
        val_dir = os.path.join(model_dir, subdir, 'val')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Look for the train folder as specified
        train_folder = os.path.join(voc_dir, subdir, 'train')
        
        if not os.path.isdir(train_folder):
            print(f"Could not find train folder in {subdir}. Skipping.")
            continue
        
        # Images are in {subdir}/train/images
        images_dir = os.path.join(train_folder, 'images')
        
        # Annotations are in the same train folder, check for common annotation folder names
        annotation_dir = None
        for anno_dirname in ['annotations', 'annotation', 'Annotations', 'xmls']:
            potential_dir = os.path.join(train_folder, anno_dirname)
            if os.path.isdir(potential_dir):
                annotation_dir = potential_dir
                break
        
        if not os.path.isdir(images_dir) or not annotation_dir:
            print(f"Could not find images or annotations directories in {train_folder}. Skipping.")
            continue
            
        print(f"Found images in: {images_dir}")
        print(f"Found annotations in: {annotation_dir}")
        
        # Find XML files
        xml_pattern = os.path.join(annotation_dir, '**', '*.xml')
        xml_files = glob.glob(xml_pattern, recursive=True)
        
        if not xml_files:
            print(f"No XML files found in {annotation_dir}. Skipping.")
            continue
            
        print(f"Found {len(xml_files)} XML files in {annotation_dir}")
        
        # Shuffle and split data
        random.shuffle(xml_files)
        train_count = int(len(xml_files) * (1 - val_split))
        train_files = xml_files[:train_count]
        val_files = xml_files[train_count:]
        
        # Process training data
        process_set(train_files, images_dir, train_dir, "training")
        
        # Process validation data
        process_set(val_files, images_dir, val_dir, "validation")
        
        print(f"Successfully processed {subdir}: {len(train_files)} training files, {len(val_files)} validation files")
        
def process_set(xml_files, images_dir, output_dir, set_name):
    """Process a set of XML files and convert them to YOLO format."""
    for xml_file in xml_files:
        # Get base filename without extension
        base_name = os.path.basename(xml_file)
        name_without_ext = os.path.splitext(base_name)[0]
        
        # Try different image extensions
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(images_dir, name_without_ext + ext)
            if os.path.exists(img_path):
                image_path = img_path
                break
        
        if not image_path:
            print(f"Warning: No matching image for {base_name}, skipping...")
            continue
        
        # Create YOLO format label
        boxes = parse_voc_annotation(xml_file)
        if not boxes:
            print(f"Warning: No valid boxes found in {base_name}, skipping...")
            continue
            
        # Copy image to output directory
        image_filename = os.path.basename(image_path)
        output_image_path = os.path.join(output_dir, image_filename)
        shutil.copy(image_path, output_image_path)
        
        # Save label file with same name but .txt extension
        label_filename = name_without_ext + '.txt'
        label_path = os.path.join(output_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for box in boxes:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert VOC format datasets to YOLOv8 format')
    parser.add_argument('--voc', type=str, default='voc', help='Path to VOC datasets directory')
    parser.add_argument('--model', type=str, default='model', help='Path to output model directory')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio (0-1)')
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    voc_dir = os.path.join(project_dir, args.voc)
    model_dir = os.path.join(project_dir, args.model)
    
    print(f"VOC directory: {voc_dir}")
    print(f"Model directory: {model_dir}")
    
    convert_to_yolo(voc_dir, model_dir, args.val_split)
    
    print("\nConversion complete! You can now use the datasets with YOLOv8.")
    print("Make sure to update your YAML configuration files to point to the new directory structure.")
