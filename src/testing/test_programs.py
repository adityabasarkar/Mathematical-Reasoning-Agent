import math
# [1] [4] Calculate angle B in radians using its sine value.
sin_B = 3/5
angle_B_rad = math.asin(sin_B)
# [2] [1] Calculate the second possible value for angle B in radians.
angle_B_rad_2 = math.pi - angle_B_rad

# [3] [2,3,6] Calculate the lengths of side BC for both possible triangles using the Law of Sines.
AB = 10
AC = 15  # Example value for b, which is greater than 10

# For the first triangle (acute angle C)
sin_C = (AB / AC) * sin_B
angle_C_rad = math.asin(sin_C)
BC_1 = AB * math.sin(angle_B_rad) / sin_C

# For the second triangle (obtuse angle C)
angle_C_rad_2 = math.pi - angle_C_rad
BC_2 = AB * math.sin(angle_B_rad_2) / math.sin(angle_C_rad_2)

# [4] [3] Calculate the positive difference between the lengths of side BC in the two triangles.
difference_BC = abs(BC_2 - BC_1)
print(difference_BC)