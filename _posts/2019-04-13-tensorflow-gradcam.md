---
layout: post
title:  "Guided GradCAM in TensorFlow"
tag:
- Deep Learning
- Tensorflow
category: blog
author: franciscosalgado
---

# Guided GradCAM in TensorFlow
Last year, I was working on optimizing a person re-identification architecture that used a CNN for feature extraction for my master thesis. I relied on several visualization techniques for understanding the descriptors being computed such as displaying the learned filters on early convolutional layers or the images that maximally activate each of the filters on the last convolutional layer. However, these only gave me a very broad idea of what was triggering each filter and I wanted something more insightful. 

Several CNN visualization techniques have been developed and a comprehensive list can be found in the [Stanford CS231n webpage](http://cs231n.github.io/understanding-cnn/). After coming across the beatufiful visualizations shown in the Guided GradCAM [1] paper, I decided to implement it myself so I could use it in my future projects.
I have posted the code alongside with the implementation and usage details in [my github](https://github.com/d3rezz/tf-guided-gradcam). Although the example code uses TensorFlow 1.13, the provided funtion to compute the Guided GradCam  maps `gradcam()` is compatible with any machine learning framework. 

GradCAM helps vizualing which parts of an input image trigger the predicted class, by backpropagating the gradients to the last convolutional layer, producing a coarse heatmap.
Guided GradCAM is then obtained by fusing GradCAM with Guided Backpropagation via element-wise multiplication, and results in a heatmap highliting much finer details. Please refer to the original paper [1] for further details on the algorithm.

Below you can see some examples:


<div id="panda-table">
    <table align="center">
	    <tr>
            <td style="padding:5px">
        	    <img src="/assets/post_images/2019-04-13-tensorflow-gradcam/panda.png" height="150" width="150" />
      	    </td>
            <td style="padding:5px">
        	    <img src="/assets/post_images/2019-04-13-tensorflow-gradcam/gradcam_panda.jpg" height="150" width="150" />
      	    </td>
    	    <td style="padding:5px">
        	    <img src="/assets/post_images/2019-04-13-tensorflow-gradcam/gbp_panda.jpg" height="150" width="150" />
      	    </td>
            <td style="padding:5px">
            	<img src="/assets/post_images/2019-04-13-tensorflow-gradcam/ggc_panda.jpg" height="150" width="150" />
             </td>
        </tr>
        <tr>
            <td align="center" style="border-top: none;">Predicted: Panda</td>
            <td align="center" >GradCAM</td>
            <td align="center" >Guided Backprop</td>
            <td align="center" >Guided GradCAM</td>
        </tr>
    </table>
</div>

<div id="car-table">
    <table align="center">
	    <tr>
            <td style="padding:5px">
        	    <img src="/assets/post_images/2019-04-13-tensorflow-gradcam/car.jpg" height="150" width="150" >
      	    </td>
            <td style="padding:5px">
        	    <img src="/assets/post_images/2019-04-13-tensorflow-gradcam/gradcam_car.jpg" height="150" width="150" >
      	    </td>
    	    <td style="padding:5px">
        	    <img src="/assets/post_images/2019-04-13-tensorflow-gradcam/gbp_car.jpg" height="150" width="150" >
      	    </td>
            <td style="padding:5px">
            	<img src="/assets/post_images/2019-04-13-tensorflow-gradcam/ggc_car.jpg" height="150" width="150" >
             </td>
        </tr>
        <tr>
            <td align="center" >Predicted: Race Car</td>
            <td align="center" >GradCAM</td>
            <td align="center" >Guided Backprop</td>
            <td align="center" >Guided GradCAM</td>
        </tr>
    </table>
</div>

By specifying the target class, it is possible to see which areas of the image contribute to it vs the predicted class:

<div id="dog-table">
    <table align="center">
	    <tr>
            <td style="padding:5px">
        	    <img src="/assets/post_images/2019-04-13-tensorflow-gradcam/demo.png" height="150" width="150" >
      	    </td>
            <td style="padding:5px">
        	    <img src="/assets/post_images/2019-04-13-tensorflow-gradcam/gradcam_dog.jpg" height="150" width="150" >
      	    </td>
    	    <td style="padding:5px">
        	    <img src="/assets/post_images/2019-04-13-tensorflow-gradcam/gbp_dog.jpg" height="150" width="150" >
      	    </td>
            <td style="padding:5px">
            	<img src="/assets/post_images/2019-04-13-tensorflow-gradcam/ggc_dog.jpg" height="150" width="150" >
             </td>
        </tr>
        <tr>
            <td align="center" >Predicted: Bulldog</td>
            <td align="center" >GradCAM</td>
            <td align="center" >Guided Backprop</td>
            <td align="center" >Guided GradCAM</td>
        </tr>
    </table>
</div>

<div id="cat-table">
    <table align="center">
	    <tr>
            <td style="padding:5px">
        	    <img src="/assets/post_images/2019-04-13-tensorflow-gradcam/demo.png" height="150" width="150" >
      	    </td>
            <td style="padding:5px">
        	    <img src="/assets/post_images/2019-04-13-tensorflow-gradcam/gradcam_cat.jpg" height="150" width="150" >
      	    </td>
    	    <td style="padding:5px">
        	    <img src="/assets/post_images/2019-04-13-tensorflow-gradcam/gbp_cat.jpg" height="150" width="150" >
      	    </td>
            <td style="padding:5px">
            	<img src="/assets/post_images/2019-04-13-tensorflow-gradcam/ggc_cat.jpg" height="150" width="150" >
             </td>
        </tr>
        <tr>
            <td align="center" >Target: Tabby cat</td>
            <td align="center" >GradCAM</td>
            <td align="center" >Guided Backprop</td>
            <td align="center" >Guided GradCAM</td>
        </tr>
    </table>
</div>

## Code
[https://github.com/d3rezz/tf-guided-gradcam](https://github.com/d3rezz/tf-guided-gradcam)

## References
[1] Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE International Conference on Computer Vision. 2017.