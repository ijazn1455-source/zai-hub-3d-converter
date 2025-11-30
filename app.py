# app.py
import gradio as gr
import torch
import cv2
import numpy as np
import open3d as o3d
from moviepy.editor import ImageSequenceClip
from diffusers import StableDiffusionPipeline

# ---------------------------------------------------
# 1. TEXT → IMAGE
# ---------------------------------------------------
def text_to_image(prompt, save_path="generated.png"):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    img = pipe(prompt).images[0]
    img.save(save_path)
    return save_path

# ---------------------------------------------------
# 2. IMAGE → DEPTH (MiDaS)
# ---------------------------------------------------
def image_to_depth(image_path):
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb).to("cuda")

    with torch.no_grad():
        depth = midas(input_batch)

    depth = depth.squeeze().cpu().numpy()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = depth.astype(np.uint8)

    cv2.imwrite("depth.png", depth)
    return depth, img

# ---------------------------------------------------
# 3. DEPTH → POINT CLOUD
# ---------------------------------------------------
def depth_to_pointcloud(depth, img):
    h, w = depth.shape
    fx = fy = 1
    cx, cy = w / 2, h / 2

    points, colors = [], []

    for y in range(h):
        for x in range(w):
            z = depth[y, x] / 255.0
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            points.append([X, Y, z])
            colors.append(img[y, x] / 255.0)

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud("model_3d.ply", pc)
    return pc

# ---------------------------------------------------
# 4. POINT CLOUD → MESH
# ---------------------------------------------------
def pointcloud_to_mesh(pc):
    pc.estimate_normals()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8)[0]
    mesh = mesh.filter_smooth_simple(number_of_iterations=2)
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh("model_3d.obj", mesh)
    return mesh

# ---------------------------------------------------
# 5. 3D ROTATING VIDEO (360°)
# ---------------------------------------------------
def create_rotation_video(mesh):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)

    images = []

    for angle in range(0, 360, 3):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(False)
        img = (np.asarray(img) * 255).astype(np.uint8)
        images.append(img)

    vis.destroy_window()

    clip = ImageSequenceClip(images, fps=30)
    clip.write_videofile("3d_rotation.mp4", codec="libx264")

# ----------------------------------------
# MAIN PROCESS FUNCTION
# ----------------------------------------
def convert_to_3d(input_type, text_prompt, image_file):
    try:
        if input_type == "Text":
            if not text_prompt:
                return None, None, "Error: Please enter a text prompt."
            img_path = text_to_image(text_prompt)

        else:
            if not image_file:
                return None, None, "Error: Please upload an image."
            img_path = image_file.name

        # PROCESS
        depth, img = image_to_depth(img_path)
        point_cloud = depth_to_pointcloud(depth, img)
        mesh = pointcloud_to_mesh(point_cloud)
        create_rotation_video(mesh)

        return "model_3d.obj", "3d_rotation.mp4", "Success! 3D model generated."

    except Exception as e:
        return None, None, f"Error: {str(e)}"

# ----------------------------------------
# GRADIO WEB INTERFACE
# ----------------------------------------
def toggle_inputs(input_type):
    if input_type == "Text":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

with gr.Blocks(title="Z@AI Hub - 3D Converter") as app:

    gr.Markdown("""
    # 🎨 Z@AI Hub – Text/Image ➜ 3D Converter  
    Convert text descriptions or images into a **3D model and 3D animation**.
    """)

    # INPUT TYPE
    input_type = gr.Radio(["Text", "Image"], label="Choose Input Type", value="Text")

    # TEXT + IMAGE INPUTS
    text_input = gr.Textbox(label="Enter Text Prompt", placeholder="A futuristic car, golden dragon, etc.")
    image_input = gr.File(label="Upload Image", visible=False)

    # Update visibility based on type
    input_type.change(toggle_inputs, inputs=input_type, outputs=[text_input, image_input])

    # BUTTON
    convert_btn = gr.Button("Convert to 3D")

    # OUTPUTS
    obj_output = gr.File(label="Download 3D Model (.obj)")
    video_output = gr.Video(label="3D Rotation Video")
    status = gr.Textbox(label="Status Message")

    # CONNECT FUNCTION
    convert_btn.click(
        convert_to_3d,
        inputs=[input_type, text_input, image_input],
        outputs=[obj_output, video_output, status]
    )

# RUN APP
app.launch()
