import bpy
import math
import mathutils

bpy.context.area.ui_type = 'VIEW_3D'
piece = bpy.context.object

counter = 0



for tilt in range(0, 4):
    for yaw in range(0, 4):
        for roll in range(0, 23):
            counter += 1
            piece.rotation_euler = [math.radians(yaw*90), math.radians(tilt*90), math.radians(roll*15)]
            piece.keyframe_insert(data_path = 'rotation_euler', frame = counter*15)

bpy.context.area.ui_type = 'DOPESHEET'
bpy.ops.action.interpolation_type(type='CONSTANT')

bpy.context.area.ui_type = 'TEXT_EDITOR'
