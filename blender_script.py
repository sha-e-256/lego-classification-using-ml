import bpy
import math
import random

bpy.context.area.ui_type = 'VIEW_3D'
piece = bpy.data.objects['Piece']
piece_mat = bpy.data.materials.new(name='piece')

plane = bpy.data.objects['Plane']
plane_mat = bpy.data.materials.new(name='plane')

num_x_rotations = 8
num_y_rotations = 8
num_z_rotations = 24
angle_interval_x = 360 / num_x_rotations
angle_interval_y = 360 / num_y_rotations
angle_interval_z = 360 / num_z_rotations

count = 0
for x in range(num_x_rotations):
    for y in range(num_y_rotations):
        for z in range(num_z_rotations):
            R1 = random.uniform(0.95, 1)
            G1 = random.uniform(0, 0.1)
            B1 = random.uniform(0, 0.1)

            R2 = random.uniform(0.95, 1)
            G2 = random.uniform(0.95, 1)
            B2 = random.uniform(0.95, 1)

            piece.active_material = piece_mat
            piece_mat.diffuse_color = (R1, G1, B1, 1)
            piece_mat.keyframe_insert(data_path="diffuse_color", frame=count)
            piece.rotation_euler = [math.radians(x * angle_interval_x), math.radians(y * angle_interval_y),
                                    math.radians(z * angle_interval_z)]
            piece.keyframe_insert(data_path='rotation_euler', frame=count)

            plane.active_material = plane_mat
            plane_mat.diffuse_color = (R2, G2, B2, 1)
            plane_mat.keyframe_insert(data_path="diffuse_color", frame=count)
            count += 1

bpy.context.area.ui_type = 'TEXT_EDITOR'
