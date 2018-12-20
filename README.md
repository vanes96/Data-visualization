# Algorithm of image binarization based on use of pyramidal threshold map

## 1. Creation of pyramidal decomposition
3 pyramids are created at this step. They are filled by local maximus, minumums and average values inside box 2x2 at each level. 0 level is equal to original image size. Each following level is 2 times smaller than previous. The lowest level is equal or bigger than 2x2.

## 2. Calculation of noise threshold [optional]

## 3. Creation of threshold map
1) At 0 level map is initialized as the pyramid of average values or each element is equal to average between local maximum and minimum from the corresponding pyramids.
2) Next level is increased by 2 times. Following convolutions are applied: [1 3] [3 1] horizontally and vertically.

## 4. Binarization using pixel thresholds from map
