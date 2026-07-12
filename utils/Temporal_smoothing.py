from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Track:
    bbox: list[float]
    confidence: float
    class_id: int
    missed :int =0

class TemporalSmoothing:
    def __init__(self,alpha:float=0.4,iou_threshold:float=0.3,max_missed:int=3):
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

        self.tracks: Dict[int, Track] ={}
        self.next_track_id =0

    def smooth(self, detections: List[Dict])->List[Dict]:
        matched_tracks =set()
        smoothed =[]

        for detection in detections:
            track_id = self.find_best_track(detection)

            if track_id is None:
                track_id =self.create_track(detection)
            else:
                self.update_track(track_id,detection)

            matched_tracks.add(track_id)
            track= self.tracks[track_id]
            smoothed.append({
                "bbox" : track.bbox.copy(),
                "confidence" :track.confidence,
                "class_id": track.class_id
            })

        remove =[]
        for track_id,track in self.tracks.items():
            if track_id not in matched_tracks:
                track.missed +=1
                if track.missed >= self.max_missed:
                    remove.append(track_id)
        for track_id in remove:
            del self.tracks[track_id]

        return smoothed 

    def create_track(self,detection:Dict)->int:
        track_id = self.next_track_id
        self.next_track_id +=1

        new_track = Track(
            bbox=detection["bbox"].copy(),
            confidence=detection["confidence"],
            class_id=detection["class_id"],
            missed=0
        )
        self.tracks[track_id]= new_track
        return track_id

    def update_track(self,track_id:int,detection:Dict):
        track = self.tracks[track_id]
        new_box =[]
        for old , new in zip(track.bbox,detection["bbox"]):
            smoothed_value = self.alpha * new + (1-self.alpha)*old
            new_box.append(smoothed_value)
        track.bbox = new_box
        track.confidence = ( self.alpha * detection["confidence"] + (1-self.alpha)*track.confidence)
        track.missed = 0

    def find_best_track(self,detection:Dict)->int:
        best_iou=0.0
        best_track=None

        for track_id,track in self.tracks.items():
            if track.class_id != detection["class_id"]:
                continue
            iou = self.calculate_iou(track.bbox,detection["bbox"])
            if iou > best_iou and iou >= self.iou_threshold:
                best_iou = iou
                best_track = track_id

        return best_track

    @staticmethod
    def calculate_iou(box1:List[float],box2:List[float])->float:
        x1 = max(box1[0],box2[0])
        y1 = max(box1[1],box2[1])
        x2 = min(box1[2],box2[2])
        y2 = min(box1[3],box2[3])

        w=max(0,x2-x1)
        h=max(0,y2-y1)
        inter_area = w * h
        box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
        box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])

        union_area = box1_area + box2_area - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area
