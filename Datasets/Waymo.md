## Waymo Open Dataset And Comparison

Links: [[Official Site]](https://waymo.com/open/about) [[GitHub]](https://github.com/waymo-research/waymo-open-dataset) [[Paper]](https://arxiv.org/pdf/1912.04838.pdf)

### Ground Truth

- 11.8M 2D bounding box labels with tracking IDs on camera data(with more than 58k unique pedestrians)
- 12.6M 3D bounding box labels with tracking IDs on lidar data( with more than 23k unique pedestrians)
- 1000+ segmentation labels for 2D and 3D data
- For key points, there are labels for 2 classes: cyclists and pedestrians:
  - 14 key points from nose to ankle
  - 200k object frames with 2D key point labels
  - 10k object frames with 3D key point labels(association provided)

### Statistics:

| Dataset |                        Format                        |          Data           | Unique Pedestrians |     Pose level      | Behaviour | Intention |
| :-----: | :--------------------------------------------------: | :---------------------: | :----------------: | :-----------------: | --------- | --------- |
|  JAAD   |        Video clips(images) +Text -Annotations        |   346 videos(1 hour)    |        686         |      2D Boxes       | Yes       | No        |
|   PIE   |      Video clips(images) + Box&Text Annotations      |      Over 6 hours       |        1.8K        |      2D Boxes       | Yes       | Yes       |
|   WOD   | Images+Point Cloud(BEV)+ Map + Box&Text Annotations  |        6.4 hours        |  23K(3D)/58K(2D)   |    3D Key points    | No        | No        |
|  TITAN  |      Video clips(images) + Box&Text Annotations      |   700 videos(10-20s)    |       395770       |      3D Boxes       | Yes       | No        |
|  LOKI   | Images+ Map(BEV) + Point Cloud+ Box&Text Annotations |      644X12.6 secs      |        <28K        |      3D Boxes       | Yes       | Yes       |
|  STIP   |            Images + Box&Text Annotations             | 15 hours(556 Scenarios) |        25K         |      2D Boxes       | No        | Yes       |
|  PedX   | Images + Point Cloud+ Segmentation&Text Annotations  |            \            |      14, 000       | 3D Key points+ mesh | No        | No        |

