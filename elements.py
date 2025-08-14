class DetectedElement:
    def __init__(self, class_id, class_name, bbox, confidence, color=None, text=None):
        self.class_id = class_id
        self.class_name = class_name
        self.bbox = bbox
        self.confidence = confidence
        self.color = color
        self.text = text 

    def __repr__(self):
        return f"<{self.class_name} conf={self.confidence:.2f} bbox={self.bbox}>"

class Transformer(DetectedElement):
    pass

class CircuitBreaker(DetectedElement):
    pass

class Switch(DetectedElement):
    pass

class MVLine(DetectedElement):
    pass