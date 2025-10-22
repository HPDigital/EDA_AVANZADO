"""
Memory management system for CAF Dashboard
Handles memory cleanup and optimization
"""
import gc
import psutil
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import weakref

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memory: float
    available_memory: float
    used_memory: float
    memory_percent: float
    python_objects: int
    garbage_objects: int


class MemoryManager:
    """Manages memory usage and cleanup"""
    
    def __init__(self, max_memory_percent: float = 80.0):
        self.max_memory_percent = max_memory_percent
        self.tracked_objects: List[weakref.ref] = []
        self.cleanup_count = 0
        self.logger = logging.getLogger(f"{__name__}.MemoryManager")
    
    def track_object(self, obj: Any) -> weakref.ref:
        """Track an object for cleanup"""
        ref = weakref.ref(obj, self._object_deleted)
        self.tracked_objects.append(ref)
        return ref
    
    def _object_deleted(self, ref):
        """Called when a tracked object is deleted"""
        if ref in self.tracked_objects:
            self.tracked_objects.remove(ref)
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Python objects
        python_objects = len(gc.get_objects())
        garbage_objects = len(gc.garbage)
        
        return MemoryStats(
            total_memory=system_memory.total / (1024**3),  # GB
            available_memory=system_memory.available / (1024**3),  # GB
            used_memory=memory_info.rss / (1024**3),  # GB
            memory_percent=system_memory.percent,
            python_objects=python_objects,
            garbage_objects=garbage_objects
        )
    
    def is_memory_high(self) -> bool:
        """Check if memory usage is high"""
        stats = self.get_memory_stats()
        return stats.memory_percent > self.max_memory_percent
    
    def cleanup_memory(self, force: bool = False) -> bool:
        """Clean up memory"""
        if not force and not self.is_memory_high():
            return False
        
        self.logger.info("Starting memory cleanup...")
        
        # Clean up tracked objects
        self._cleanup_tracked_objects()
        
        # Force garbage collection
        collected = 0
        for i in range(3):
            collected += gc.collect()
        
        self.cleanup_count += 1
        
        # Log results
        stats = self.get_memory_stats()
        self.logger.info(
            f"Memory cleanup completed. "
            f"Collected {collected} objects. "
            f"Memory usage: {stats.memory_percent:.1f}%"
        )
        
        return True
    
    def _cleanup_tracked_objects(self):
        """Clean up tracked objects"""
        # Remove dead references
        self.tracked_objects = [ref for ref in self.tracked_objects if ref() is not None]
        
        # Call cleanup methods on tracked objects
        for ref in self.tracked_objects:
            obj = ref()
            if obj is not None and hasattr(obj, 'cleanup'):
                try:
                    obj.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up object: {str(e)}")
    
    def cleanup_object(self, obj: Any):
        """Clean up a specific object"""
        if hasattr(obj, 'cleanup'):
            try:
                obj.cleanup()
                self.logger.debug(f"Cleaned up object: {type(obj).__name__}")
            except Exception as e:
                self.logger.warning(f"Error cleaning up object: {str(e)}")
        else:
            # Try to delete attributes
            if hasattr(obj, '__dict__'):
                for attr_name in list(obj.__dict__.keys()):
                    try:
                        delattr(obj, attr_name)
                    except:
                        pass
    
    def get_cleanup_count(self) -> int:
        """Get number of cleanup operations performed"""
        return self.cleanup_count
    
    def reset_cleanup_count(self):
        """Reset cleanup counter"""
        self.cleanup_count = 0
    
    def force_cleanup(self):
        """Force immediate cleanup"""
        self.cleanup_memory(force=True)
    
    def monitor_memory(self, callback=None):
        """Monitor memory usage and cleanup if needed"""
        if self.is_memory_high():
            self.cleanup_memory()
            if callback:
                callback(self.get_memory_stats())
