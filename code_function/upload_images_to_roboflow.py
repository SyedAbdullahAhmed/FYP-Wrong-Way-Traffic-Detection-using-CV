import os
from roboflow import Roboflow

def upload_images_to_roboflow(image_folder, api_key, workspace_name, project_name):
    """
    Uploads images from a folder to a Roboflow project.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_name)

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Uploading {image_name}...")
            try:
                project.upload(image_path)
                print(f"Uploaded {image_name} successfully.")
            except Exception as e:
                print(f"Failed to upload {image_name}: {e}")
                return f"Failed to upload {image_name}: {e}"

    return "All images uploaded successfully!"