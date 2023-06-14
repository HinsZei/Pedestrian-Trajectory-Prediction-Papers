## Waymo Open Dataset

Links: [[Official Site]](https://waymo.com/open/about) [[GitHub]](https://github.com/waymo-research/waymo-open-dataset) [[Paper]](https://arxiv.org/pdf/1912.04838.pdf)

### Ground Truth

- 11.8M 2D bounding box labels with tracking IDs on camera data(with more than 58k unique pedestrians)
- 12.6M 3D bounding box labels with tracking IDs on lidar data( with more than 23k unique pedestrians)
- 1000+ segmentation labels for 2D and 3D data
- For key points, there are labels for 2 classes: cyclists and pedestrians:
  - 14 key points from nose to ankle
  - 200k object frames with 2D key point labels
  - 10k object frames with 3D key point labels(association provided)



| Dataset |                   Format                   | Data               | Unique Pedestrians | Pose level | Behaviour | Intention |
| :-----: | :----------------------------------------: | ------------------ | ------------------ | ---------- | --------- | --------- |
|  JAAD   |   Video clips(images) +Text Annotations    | 346 videos(1 hour) | 686                | Boxes      | Yes       | No        |
|   PIE   | Video clips(images) + Box&Text Annotations | Over 6 hours       | 1.8K               | Boxes      | Yes       | Yes       |
|   WOD   |     Images+Lidar+ Box&Text Annotations     | 6.4 hours          | 23K(3D)/58K(2D)    | Key points | No        | No        |

