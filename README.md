Dataset
Leafsnap Dataset
Source: http://leafsnap.com/dataset/
The dataset used for this project is comprised of two collections of leaf images from 184 distinct tree species that inhabit the northeastern United States. 
Collection #1: 23147 lab images of high quality; photos taken of pressed leaves. These images are from a Smithsonian collection and were taken in a controlled environment with back-lit and front-lit versions; includes an average of ~126 samples per species.
Collection #2: 7719 field images of variable quality, representative of the type of photo which would be taken by mobile devices (most by iPhone) and in an outdoor environment. Images contain some combination of blur, illumination patterns, shadows, and other random “noise”; includes an average of ~42 samples per species.

The computer utilized for this project has the following specifications:
•	Chip - Apple M3 Max
o	14 core CPU
o	30 core GPU
o	16 core Neural Engine hardware block dedicated for ML tasks (only useful when using Apple’s software stack).
•	RAM - 36 GB
•	Operating System – macOS Tahoe v. 26.3.1
Project code specifies that Apple’s Metal Performance Shaders (MPS) should be used, which routes the training computations to the GPU.

Model Summary:
 
<img width="468" height="515" alt="image" src="https://github.com/user-attachments/assets/d787d91c-ca05-47c2-b8d7-7af5caaa06f4" />

