import cv2
import numpy as np
import os
import gradio as gr
import json
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure


class TextureComponentExtractor:
    def __init__(self, input_image_path, output_dir="output", min_component_size=100):
        self.input_image_path = input_image_path
        self.output_dir = output_dir
        self.min_component_size = min_component_size

        self.components_dir = os.path.join(output_dir, "components")
        self.reassembled_dir = os.path.join(output_dir, "reassembled")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.components_dir, exist_ok=True)
        os.makedirs(self.reassembled_dir, exist_ok=True)

        self.image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image: {input_image_path}")

        self.height, self.width = self.image.shape[:2]
        self.has_alpha = self.image.shape[2] == 4 if len(self.image.shape) == 3 else False

        self.components = []
        self.component_metadata = {}

    def extract_components(self):
        if self.has_alpha:
            alpha = self.image[:, :, 3]
            mask = (alpha > 10).astype(np.uint8)
        elif len(self.image.shape) > 2:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        else:
            _, mask = cv2.threshold(self.image, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create bounding boxes
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < self.min_component_size:
                continue
            bboxes.append((x, y, x + w, y + h))

        # Group overlapping bounding boxes
        groups = []
        visited = [False] * len(bboxes)

        def boxes_overlap(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

        def dfs(idx, group):
            visited[idx] = True
            group.append(bboxes[idx])
            for i, box in enumerate(bboxes):
                if not visited[i] and boxes_overlap(bboxes[idx], box):
                    dfs(i, group)

        grouped_boxes = []
        for i in range(len(bboxes)):
            if not visited[i]:
                group = []
                dfs(i, group)
                # Merge group into one bounding box
                xs = [b[0] for b in group] + [b[2] for b in group]
                ys = [b[1] for b in group] + [b[3] for b in group]
                grouped_boxes.append((min(xs), min(ys), max(xs), max(ys)))

        # Now crop using merged boxes
        pad = 5
        component_id = 1
        self.components = []
        self.component_metadata = {}

        for x1, y1, x2, y2 in grouped_boxes:
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(self.width, x2 + pad)
            y2 = min(self.height, y2 + pad)

            component = self.image[y1:y2, x1:x2]
            component_name = f"component_{component_id}"
            component_id += 1

            component_info = {
                "id": component_id - 1,
                "name": component_name,
                "bbox": [y1, x1, y2, x2],
                "position": [x1, y1],
                "size": [x2 - x1, y2 - y1],
                "file": f"{component_name}.png"
            }

            self.component_metadata[component_name] = component_info
            self.components.append(component_info)

            component_path = os.path.join(self.components_dir, f"{component_name}.png")
            if self.has_alpha:
                cv2.imwrite(component_path, component)
            else:
                component_rgb = cv2.cvtColor(component, cv2.COLOR_BGR2RGB)
                Image.fromarray(component_rgb).save(component_path)

        metadata_path = os.path.join(self.output_dir, "components_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "image_size": [self.width, self.height],
                "components": self.components
            }, f, indent=2)

    def visualize_components(self):
        if self.has_alpha:
            visual_img = cv2.cvtColor(self.image[:, :, :3], cv2.COLOR_BGR2RGB)
        else:
            visual_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        for component in self.components:
            min_row, min_col, max_row, max_col = component["bbox"]
            cv2.rectangle(visual_img, (min_col, min_row), (max_col, max_row), (0, 255, 0), 2)
            cv2.putText(visual_img, f"{component['id']}", (min_col, min_row - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        viz_path = os.path.join(self.output_dir, "components_visualization.png")
        Image.fromarray(visual_img).save(viz_path)
        return viz_path

    def reassemble_components(self):
        if self.has_alpha:
            reassembled = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        else:
            reassembled = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for component in self.components:
            component_name = component["name"]
            component_path = os.path.join(self.components_dir, f"{component_name}.png")

            comp_img = cv2.imread(component_path, cv2.IMREAD_UNCHANGED)
            if comp_img is None:
                continue

            min_row, min_col, max_row, max_col = component["bbox"]

            if self.has_alpha and comp_img.shape[2] == 4:
                alpha = comp_img[:, :, 3] / 255.0
                alpha = np.expand_dims(alpha, axis=2)
                for c in range(3):
                    reassembled[min_row:max_row, min_col:max_col, c] = (
                        comp_img[:, :, c] * alpha[:, :, 0]
                    ).astype(np.uint8)
                reassembled[min_row:max_row, min_col:max_col, 3] = comp_img[:, :, 3]
            else:
                if len(comp_img.shape) == 3 and comp_img.shape[2] >= 3:
                    h, w = comp_img.shape[:2]
                    if min_row + h <= self.height and min_col + w <= self.width:
                        reassembled[min_row:min_row + h, min_col:min_col + w] = comp_img[:, :, :3]

        output_path = os.path.join(self.reassembled_dir, "reassembled_texture.png")
        cv2.imwrite(output_path, reassembled)
        return output_path

    def get_component_image(self, idx):
        comp = self.components[idx]
        path = os.path.join(self.components_dir, comp["file"])
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def draw_overlay_with_red_box(self, index):
        viz_path = os.path.join(self.output_dir, "components_visualization.png")
        img = cv2.imread(viz_path)
        comp = self.components[index]
        min_row, min_col, max_row, max_col = comp["bbox"]
        cv2.rectangle(img, (min_col, min_row), (max_col, max_row), (0, 0, 255), 3)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def launch_texture_editor_workflow(input_image_path):
    extractor = TextureComponentExtractor(input_image_path)
    

    def extract():
        extractor.extract_components()
        extractor.visualize_components()
        return f"Extracted {len(extractor.components)} components.", gr.update(value=0)

    def reassemble():
        return extractor.reassemble_components()

    def get_component_data(idx):
        img = extractor.get_component_image(idx)
        overlay = extractor.draw_overlay_with_red_box(idx)
        return img, overlay, f"Component {idx + 1}/{len(extractor.components)}"

    def update_label(new_name, idx):
        idx = int(idx)
        component = extractor.components[idx]
        old_name = component["name"]
        if new_name and new_name != old_name:
            old_path = os.path.join(extractor.components_dir, f"{old_name}.png")
            new_path = os.path.join(extractor.components_dir, f"{new_name}.png")
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
            component["name"] = new_name
            component["file"] = f"{new_name}.png"
        return f"Renamed to {new_name}"

    def next_component(idx):
        return min(idx + 1, len(extractor.components) - 1)

    def prev_component(idx):
        return max(idx - 1, 0)

    with gr.Blocks() as demo:
        current_index = gr.State(0)
        gr.Markdown("# Texture Component Tool")
        with gr.Row():
            extract_btn = gr.Button("Extract Components")
            reassemble_btn = gr.Button("Reassemble Texture")
        extract_out = gr.Textbox(label="Extract Status")
        reassembled_out = gr.Image(label="Reassembled Texture")

        with gr.Row():
            prev_btn = gr.Button("Previous")
            next_btn = gr.Button("Next")
            name_box = gr.Textbox(label="Rename Component")
            rename_btn = gr.Button("Rename")

        info = gr.Textbox(label="Component Info", interactive=False)
        comp_img = gr.Image(label="Component Image")
        overlay = gr.Image(label="Overlay with Red Box")
        rename_out = gr.Textbox(label="Rename Output")

        extract_btn.click(fn=extract, inputs=[], outputs=[extract_out, current_index])
        reassemble_btn.click(fn=reassemble, inputs=[], outputs=reassembled_out)

        prev_btn.click(fn=prev_component, inputs=current_index, outputs=current_index)
        next_btn.click(fn=next_component, inputs=current_index, outputs=current_index)

        current_index.change(fn=get_component_data, inputs=current_index, outputs=[comp_img, overlay, info])
        rename_btn.click(fn=update_label, inputs=[name_box, current_index], outputs=rename_out)

    demo.launch()

# Example usage:
# launch_texture_editor_workflow("RACOON01/RACOON01.4096/texture_00.png")
launch_texture_editor_workflow("model/RACOON01.4096/texture_00.png")
