# Algorithm of image binarization based on use of pyramidal threshold map

## 1. Creation of pyramidal decomposition
3 pyramids are created at this step. 
They are filled by local maximus, minumums and average values inside box 2x2 at each level. 0 level is equal to original image size. The lowest level is equal or bigger than 2x2.

## 2. Calculation of noise threshold [optional]

## 3. Creation of threshold map

## 4. Binarization using pixel thresholds from map
