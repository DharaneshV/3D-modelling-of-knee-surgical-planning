import unittest
import numpy as np
import trimesh
import sys
import os
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.report_generator import get_roi_sizing, calculate_anatomic_axis

class TestReportGeneratorMetrics(unittest.TestCase):
    
    def test_jsw_non_overlapping(self):
        # Create a simple femur box and tibia box separated by 3.5mm
        # Femur box of size 20x20x10 centered at (0, 0, 55) => Z from 50 to 60
        femur = trimesh.creation.box(extents=[20, 20, 10])
        femur.apply_translation([0, 0, 55])
        
        # Tibia box of size 20x20x10 centered at (0, 0, 41.5) => Z from 36.5 to 46.5
        tibia = trimesh.creation.box(extents=[20, 20, 10])
        tibia.apply_translation([0, 0, 41.5])
        
        # Check collision
        import pyvista as pv
        pv_femur = pv.wrap(femur)
        pv_tibia = pv.wrap(tibia)
        collision, n_contacts = pv_femur.collision(pv_tibia)
        is_collision = n_contacts > 0
        
        self.assertFalse(is_collision)
        
        # Check distance
        tree = cKDTree(tibia.vertices)
        dists, _ = tree.query(femur.vertices)
        jsw = round(float(np.min(dists)), 2)
        self.assertEqual(jsw, 3.5)
        
    def test_jsw_overlapping(self):
        # Femur box of size 10x10x10 centered at (0, 0, 55) => Z from 50 to 60
        femur = trimesh.creation.box(extents=[10, 10, 10])
        femur.apply_translation([0, 0, 55])
        
        # Tibia box of size 20x20x10 centered at (0, 0, 47) => Z from 42 to 52
        tibia = trimesh.creation.box(extents=[20, 20, 10])
        tibia.apply_translation([0, 0, 47])
        
        import pyvista as pv
        pv_femur = pv.wrap(femur)
        pv_tibia = pv.wrap(tibia)
        collision, n_contacts = pv_femur.collision(pv_tibia)
        is_collision = n_contacts > 0
        
        self.assertTrue(is_collision)

    def test_pca_ml_ap_rotation(self):
        theta = np.linspace(0, 2*np.pi, 100)
        z = np.linspace(0, 30, 10)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        x = 40 * np.cos(theta_grid)
        y = 25 * np.sin(theta_grid)
        
        vertices = np.column_stack([x.flatten(), y.flatten(), z_grid.flatten()])
        long_axis = np.array([0, 0, 1])
        
        ml, ap = get_roi_sizing(vertices, long_axis, is_femur=True, side='test')
        self.assertAlmostEqual(ml, 80.0, places=1)
        self.assertAlmostEqual(ap, 50.0, places=1)
        
        angle = np.pi / 4
        rot_mat = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated_vertices = np.dot(vertices, rot_mat.T)
        
        ml_rot, ap_rot = get_roi_sizing(rotated_vertices, long_axis, is_femur=True, side='test')
        
        self.assertAlmostEqual(ml_rot, 80.0, places=1)
        self.assertAlmostEqual(ap_rot, 50.0, places=1)

if __name__ == '__main__':
    unittest.main()
