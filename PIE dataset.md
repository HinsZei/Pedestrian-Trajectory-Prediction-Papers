## PIE: Pedestrian Intention Estimation Dataset

Links: [[Official Site]](https://data.nvision2.eecs.yorku.ca/PIE_dataset/) [[GitHub]](https://github.com/aras62/PIE#clips) [[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.pdf)

### Ground Truth

over 6 hours of driving footage captured with calibrated monocular dashboard camera Waylens Horizon equipped with 157◦ wide angle lens. All videos are recorded in HD format (1920 × 1080) at 30 fps. The videos are split to10 minute long chunks, grouping into 6 sets, with 911k frames, in which 1.8k pedestrians is labelled by behaviour and intention.



(Copied from site page)Objects are annotated with bounding boxes using two-point coordinates (top-left, bottom-right) `[x1, y1, x2, y2]`. The bounding boxes for pedestrians have corresponding occlusion tags with the following numeric values:

- 0 - not occluded (pedestrian is fully visible or <25% of the bbox area is occluded);
- 1 - partially occluded (between 25% and 75% of the bbox area is occluded)
- 2 - fully occluded (>75% of the bbox area is occluded).

Other types of objects have binary occlusion labels: 0 (fully visible) or 1 (partially or fully occluded).

Depending on the type of object additional information is provided for each bounding box (where applicable):

- pedestrian - textual labels for actions, gestures, looking, or crossing
  - Action: Whether the pedestrian is `walking` or `standing`
  - Gesture: The type of gestures exhibited by the pedestrian. The gestures include
    `hand_ack` (pedestrian is acknowledging by hand gesture),`hand_yield` (pedestrian is yielding by hand gesture), `hand_rightofway` (pedestrian is giving right of way by hand gesture), `nod`, or `other`.
  - Look: Whether pedestrian is `looking` or `not-looking` in the direction of the ego-vehicle.
  - Cross: Whether pedestrian is `not-crossing`, `crossing` the path of the ego-vehicle and `crossing-irrelevant` which indicates that the pedestrian is crossing the road but not in the path of the ego-vehicle.
- vehicle
  - Type: The type of vehicle. The options are `car`, `truck`, `bus`, `train`, `bicycle` and `bike`.
- traffic_light
  - Type: The type of traffic light. The options are `regular`, `transit` (specific to buses and trains) and `pedestrian`.
  - State: The state of the traffic light. The options are `red`, `yellow` and `green`.
- sign
  - Type: The type of sign. The options are `ped_blue`, `ped_yellow`, `ped_white`, `ped_text`, `stop_sign`, `bus_stop`, `train_stop`, `construction`, `other`.
- crosswalk - none
- transit_station
  - bus or streetcar station



#### Object attributes

These include information regarding pedestrians' demographics, crossing point, crossing characteristics, etc. This information is provided for each pedestrian track:

- age: `child`, `adult` or `senior`.
- gender: `male` or `female`.
- id: Pedestrian's id.
- num_lanes: Scalar value, e.g. 2, 4, indicating the number of lanes at the point of crossing
- signalized: Indicates whether the crosswalk is signalized. Options are `n/a` (no signal, no crosswalk), `C` (crosswalk lines or pedestrian crossing sign), `S` (signal or stop sign) and `CS` (crosswalk or crossing sign combined with traffic lights or stop sign).
- traffic_direction: `OW` (one-way) or `TW` (two-way).
- intersection: Specifies the type of intersection: `midblock`, `T`, `T-right`, `T-left`, `four-way`.
- crossing: `1` (crossing), `0` (not crossing), `-1` (irrelevant). This indicates whether the pedestrian was observed crossing the road in front of the ego-vehicle. Irrelevant pedestrians are those judged as not intending to cross but standing close to the road, e.g. waiting for a bus or hailing a taxi.
- exp_start_point: The starting frame of the clip used for human experiment (see the paper for details)
- critical_point: The last frame of the clip used for human experiment
- intention_prob: A value in range `[0,1]` indicating the average human responses for the pedestrian's intention. This value is estimated intention of a given pedestrian to cross *prior to the critical point*. Therefore, there is a *single intention estimate per each pedestrian track*.
- crossing_point: The frame at which the pedestrian starts crossing. In the cases where the pedestrians do not cross the road, the last frame - 3 is selected.

**Note regarding action/intention distinction**: In the PIE dataset we distinguish between intention to cross and crossing action. We consider intention as a mental state that precedes the action but does not necessarily cause the action immediately, e.g. if it is dangerous to do so. In the case of crossing the road this leads to three possible scenarios:

- Pedestrian intends (wants) to cross and crosses because the conditions are favorable (e.g. green light for pedestrian or the ego-vehicle yields);
- Pedestrian intends to cross but cannot cross since the conditions prevent them from doing so (e.g. red light, being blocked by other pedestrians or vehicles not yielding);
- Pedestrian does not intend to cross (e.g. is talking to another person or waiting for a bus at the bus station) and therefore does not cross.

These examples illustrate that intention and action are related but are not the same. However, in the literature, these terms are often used interchangeably. Therefore, when using PIE or comparing the results of the models trained on PIE dataset it is important to understand the difference and clarify what data was used for training.

Note that the volume of training data available for action and intention is different. Action labels (moving, crossing, looking, gesturing) are provided for each bounding box in the pedestrian track (where applicable). An intention label is provided only for a set of frames *preceding the observed action* (crossing or not crossing).

In general, models that are trained on action labels will not be comparable to models trained on intention labels and will not output the same results. For example, intention estimation sub-model in [PIEPredict](https://github.com/aras62/PIEPredict) is trained on intention data (i.e. only on frames preceding the crossing action and on intention labels) and predicts intention to cross, not action. It will classify pedestrian waiting at the red traffic light as *intending to cross* even though the pedestrian is not observed crossing. An action prediction algorithm (e.g. [PCPA](https://github.com/ykotseruba/PedestrianActionBenchmark)) in this case will output *not crossing* action. Both outputs are correct but mean different things and should not be compared.



#### Ego-vehicle information

Ego-vehicle information is OBD sensor output provided per-frame. The following properties are available: `GPS_speed` (km/h), `OBD_speed` (km/h), `heading_angle`, `latitude`, `longitude`, `pitch`, `roll`, `yaw`, `acceleration` and `gyroscope`.



#### Pedestrian Intention Estimation & Trajectory Prediction

The problem have 2 levels: Early anticipation in the form of estimating pedestrians’ intention of crossing and trajectory prediction as late forecasting of the future trajectory of pedestrians based on observed scene dynamics. The former primarily serves as a refinement procedure to change the focus of an intelligent system to those pedestrians that matter, or potentially will interact with the vehicle. 

The system receives as input a sequence of images and the current speed of the ego-vehicle. The intention estimation model’s encoder receives as input a square cropped image around the pedestrians, produces some representation which is concatenated with their observed locations (bounding box coordinates) before feeding them to the decoder. The speed model predicts future speed using an encoder-decoder scheme followed by a series of self-attention units. The location prediction unit receives location information as encoder input and the combination of encoder representations, pedestrian intention and future speed as decoder input, and predicts future trajectory. **Note that they argue that the pose is implicitly encoded in the bounding box**.

![pie_predict_diagram.png](https://github.com/aras62/PIEPredict/blob/master/pie_predict_diagram.png?raw=true)