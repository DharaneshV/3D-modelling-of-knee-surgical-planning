import pyvista as pv
import os

def generate_thumbnail(case_id, out_path):
    print(f"Generating thumbnail for {case_id}...")
    mesh_path = f"outputs/{case_id}/meshes/femur_decimated.obj"
    if not os.path.exists(mesh_path):
        print(f"Mesh not found: {mesh_path}")
        return
        
    mesh = pv.read(mesh_path)
    
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color='ivory', smooth_shading=True, pbr=True, metallic=0.1, roughness=0.5)
    plotter.set_background('white')
    
    plotter.camera_position = 'yz'
    plotter.camera.azimuth = 30
    plotter.camera.elevation = 20
    plotter.camera.zoom(1.2)
    
    plotter.screenshot(out_path)
    plotter.close()
    print(f"Saved {out_path}")

cases = ['STS_006', 'STS_009', 'STS_010']
out_dir = r"C:\Users\dhara\.gemini\antigravity-ide\brain\b22556eb-ad4e-4776-96cd-98b54cc0f355\scratch"
os.makedirs(out_dir, exist_ok=True)

for case in cases:
    generate_thumbnail(case, os.path.join(out_dir, f"{case}_render.png"))
