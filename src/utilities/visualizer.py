def show(img_path: str, downsize_ratio: int = 1):
    viewer = ImageShow.IPythonViewer()
    with Image.open(img_path) as img:
        width = img.width // downsize_ratio
        height = img.height // downsize_ratio
        resized_img = img.resize((width, height))
        viewer.show(image=resized_img)