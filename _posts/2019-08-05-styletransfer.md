---
layout: post
title:  "Hip Hop album covers reimagined as famous paintings"
tag:
- Style transfer
- Deep Learning
- Tensorflow
category: blog
author: franciscosalgado
---

In Andrew Ng's Deep Learning course, students are encouraged to play with a Neural Style Transfer algorithm for one of the assignments. The assignment starts with implementing the algorithm described in "A Neural Algorithm of Artistic Style" by Gatys et al. and then students should try running it on their own style and content images.

Below are my results of applying Neural Style Transfer to the covers of some of my favourite hip hop projects, in order to imitate the style of the European painters Van Gogh, Claude Monet and Edvard Munch.

<table align="center" style="border: none; border-collapse: collapse; background-color: #ffffff;">
    <tr style="border: none; background-color: #ffffff;">
        <td align="center" style="border: none;">
            <img src="/assets/post_images/2019-08-05-styletransfer/graduation_vangogh.jpg" height="150" />
        </td>
        <td align="center" style="border: none;">
            <img src="/assets/post_images/2019-08-05-styletransfer/tyler_monet.png" height="150" />
        </td>
        <td align="center" style="border: none;">
            <img src="/assets/post_images/2019-08-05-styletransfer/chance_scream.png" height="150" />
        </td>
    </tr>
    <tr style="border: none; background-color: #ffffff;">
         <td style="border: none;" align="center">Graduation + Starry Night</td>
        <td style="border: none;" align="center">Wolf + Poppy Field</td>
        <td style="border: none;" align="center">Acid Rap + The Scream</td>
    </tr>
</table>

Tyler, the Creator's Wolf mixed with Monet's Poppy Field was my favourite result of all 3.

After trying a couple more examples, I found this algorithm very brittle: it requires a very careful selection of the hyperparameters to ensure pleasant results and reduce artifacts.
In addition, it requires running an optimization task each time which is not pratical.

In the future I would like to experiment with more robust Neural Style Transfer algorithms such as [1], which uses a style prediction network to learn "style" from an image on a single forward pass. A great blog post on porting this algorithm to run on the browser using network distillation can be found in [2].

## References
[1] Ghiasi, Golnaz, et al. "Exploring the structure of a real-time, arbitrary neural artistic stylization network." (2017).

[2] Magenta, "Porting Arbitrary Style Transfer to the Browser", 2018, https://magenta.tensorflow.org/blog/2018/12/20/style-transfer-js/ (accessed 5 Aug 2019)