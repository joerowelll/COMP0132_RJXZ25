# Copyright 2022 Joseph Rowell>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do 
# so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial 
# portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
#  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
#  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

class Line:
    def __init__(self, vector, point_on_line) -> None:
        self.vector = np.array(vector).reshape(3, 1)
        self.point_on_line = np.array(point_on_line).reshape(3, 1)
    def get_intersect(self, plane):
        if plane.normal_vector.ravel().dot(self.vector.ravel()) != 0:
            d = (plane.point_on_plane - self.point_on_line).ravel().dot(
                plane.normal_vector.ravel()) / plane.normal_vector.ravel().dot(
                    self.vector.ravel())
            return self.point_on_line + (d * self.vector)
        return None


class Plane:
    def __init__(self, normal_vector, point_on_plane) -> None:
        self.normal_vector = np.array(normal_vector).reshape(3, 1)
        self.point_on_plane = np.array(point_on_plane).reshape(3, 1)

    def intersect(self, line: Line):
        if self.normal_vector.ravel().dot(line.vector.ravel()) != 0:
            d = (self.point_on_plane - line.point_on_line).ravel().dot(
                self.normal_vector.ravel()) / self.normal_vector.ravel().dot(
                    line.vector.ravel())
            return line.point_on_line + (d * line.vector)
        return None 

def example_usage():
    #from plane_intersection import Line, Plane
    l1 = Line((1, 1, 0), (0, 0, 0))
    p1 = Plane((1, 1, 1), (5, 5, 5))
    print(p1.intersect(l1))
