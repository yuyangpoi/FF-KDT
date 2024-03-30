# Towards Robust Keypoint Detection and Tracking: A Fusion Approach with Event-Aligned Image Features



## Abstract

Robust keypoint detection and tracking are crucial for various robotic tasks. However, conventional cameras struggle under rapid motion and lighting changes, hindering local and edge feature extraction essential for keypoint detection and tracking. Event cameras offer advantages in such scenarios due to their high dynamic range and low latency. Yet, their inherent noise and motion dependence can lead to feature instability. This paper presents a novel image-event fusion approach for robust keypoint detection and tracking under challenging conditions. We leverage the complementary strengths of image and event data by introducing: (i) the Implicit Compensation Module and Temporal Alignment Module for high-frequency feature fusion and keypoint detection; and (ii) a temporal neighborhood matching strategy for robust keypoint tracking within a sliding window. Furthermore, a self-supervised temporal response consistency constraint ensures keypoint continuity and stability. Extensive experiments demonstrate the effectiveness of our method against state-of-the-art approaches under diverse challenging scenarios. Notably, our method exhibits the longest tracking lifetime and strong generalization ability on real-world data.  

![Network (1)_00](C:\Users\16523\Desktop\Network (1)_00.png)