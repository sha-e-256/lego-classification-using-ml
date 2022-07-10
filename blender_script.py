import bpy
import math
import random

bpy.context.area.ui_type = 'VIEW_3D'

light_right_data = bpy.data.lights.new('Light_right', type='POINT')
light_right = bpy.data.objects.new('Light_right', light_right_data)
bpy.context.collection.objects.link(light_right)
light_right.location = (100, 100, 400)

light_left_data = bpy.data.lights.new('Light_left', type='POINT')
light_left = bpy.data.objects.new('Light_left', light_left_data)
bpy.context.collection.objects.link(light_left)
light_left.location = (-100, -100, 400)

camera_data = bpy.data.cameras.new('Camera')
camera = bpy.data.objects.new('Camera', camera_data)
bpy.context.collection.objects.link(camera)
camera.location = (0, 0, 300)
camera.rotation_euler = (0, 0, 0)

piece_mat = bpy.data.materials.new(name='colored')
piece = bpy.data.objects['Piece']
piece.active_material = piece_mat

counter = 0

for tilt in range(0, 4):
    for yaw in range(0, 4):
        for roll in range(0, 23):
            R = random.uniform(0, 1)
            G = random.uniform(0, 1)
            B = random.uniform(0, 1)
            piece.rotation_euler = [math.radians(yaw * 90), math.radians(tilt * 90), math.radians(roll * 15)]
            piece.keyframe_insert(data_path='rotation_euler', frame=counter)

            piece_mat.diffuse_color = (R, G, B, 1)
            piece_mat.keyframe_insert(data_path="diffuse_color", frame=counter)
            counter += 1

bpy.context.area.ui_type = 'DOPESHEET'
bpy.ops.action.interpolation_type(type='CONSTANT')

bpy.context.area.ui_type = 'TEXT_EDITOR'
