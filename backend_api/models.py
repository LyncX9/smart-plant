"""
SQLAlchemy models for scan history storage.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.sql import func
from database import Base


class ScanHistory(Base):
    """Model for storing scan history."""
    __tablename__ = "scan_history"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    plant_type = Column(String(50), default="rice")
    condition = Column(String(50))  # Healthy, Diseased, Uncertain
    predicted_class = Column(String(100))  # BrownSpot, Healthy, Hispa, LeafBlast
    confidence = Column(Float)
    health_score = Column(Float)
    total_lesion_count = Column(Integer, default=0)
    avg_lesion_area_percent = Column(Float, default=0.0)
    leaves_count = Column(Integer, default=0)
    leaves_json = Column(Text)  # JSON string of leaf details
    
    def to_dict(self):
        """Convert model to dictionary for API response."""
        return {
            "id": self.id,
            "created_at": self.timestamp.isoformat() if self.timestamp else None,
            "plant_type": self.plant_type,
            "condition": self.condition,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "health_score": self.health_score,
            "total_lesion_count": self.total_lesion_count,
            "avg_lesion_area_percent": self.avg_lesion_area_percent,
            "leaves_count": self.leaves_count,
        }
    
    def to_detail_dict(self):
        """Convert model to detailed dictionary including leaves."""
        import json
        base = self.to_dict()
        base["leaves"] = json.loads(self.leaves_json) if self.leaves_json else []
        return base
