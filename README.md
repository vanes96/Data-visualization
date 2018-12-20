# Algorithm of image binarization based on use of pyramidal threshold map

## 1. Creation of pyramidal decomposition
3 pyramids are created at this step. They are filled by local maximus, minumums and average values inside box 2x2 at each level. 0 level is equal to original image size. Each following level is 2 times smaller than previous. The lowest level is equal or bigger than 2x2.

## 2. Calculation of noise threshold [optional]
Default value of brightness threshold noise is 30 (0-255).

## 3. Creation of threshold map
![Pyramids](https://github.com/vanes96/Data-visualization/blob/master/pyramids.gif)
1) At 0 level map is initialized as the pyramid of average values () or each element is equal to average between local maximum and minimum [(local_max_ij + local_min_ij) / 2] from the corresponding pyramids.
2) Next level is increased by 2 times. Following convolutions are applied: [1 3] [3 1] horizontally and vertically.
3) For each cell in thsreshold map the expression [local_max_ij - local_min_ij] is calculated. If its bigger than noise threshold the cell is defined like [local_max_ij + local_min_ij) / 2] otherwise its not changed at all.
4) Points 2 and 3 are repeated until the size of the current level of threshold_map is equal to size of the original image (0 level of each pyramid)

## 4. Binarization using pixel thresholds from map
Brightness of each pixel in original image is compared to corresponding value in threshold map: if its bigger it will become 255 (white) otherwise 0 (black) and set as pixel of new binarize image.

## Here are some examples of using algorithm:
### Original image1
![Original image2](https://github.com/vanes96/Data-visualization/blob/master/Original%20images/text3.jpg)
### Binarized image1
![Original image2](https://github.com/vanes96/Data-visualization/blob/master/Binarized%20images/text3_binarized.png)

### Original image2
![Original image1](https://github.com/vanes96/Data-visualization/blob/master/Original%20images/text2.png)
### Binarized image2
![Original image1](https://github.com/vanes96/Data-visualization/blob/master/Binarized%20images/text2_binarized.png)


