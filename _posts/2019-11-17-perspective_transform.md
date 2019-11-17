---
layout: post
title:  "Manually Computing a Perspective Transformation Matrix"
tag:
- Python
- Computer Vision
- OpenCV
category: blog
author: franciscosalgado
---

While training a Convolutional Neural Network, it might be useful to make it independent of viewpoint from which the scene was captured.  Data augmentation can be used to apply different perspective transforms to the input image.

I put together a small python function to manually compute a perspective transform matrix to be applied to an image given rotations <img src="https://latex.codecogs.com/svg.latex?%5Ctheta"/> (around the z-axis) and <img src="https://latex.codecogs.com/svg.latex?%5Cphi"/> (around the x-axis) in the camera frame. You can find it on [my github](https://github.com/d3rezz/perspective_transform).


<div align="center">
    <table align="center">
        <tr>
            <td style="padding:5px">
                <img src="/assets/post_images/2019-11-17-perspective_transform/bmw.jpg" height="250" />
            </td>
            <td style="padding:5px">
                <img src="/assets/post_images/2019-11-17-perspective_transform/transformed.jpg" height="250" />
            </td> 
        </tr>
        <tr>
            <td align="center" style="border-top: none">Original</td>
            <td align="center" style="border-top: none">Transformed</td>
        </tr>
    </table>
</div>

