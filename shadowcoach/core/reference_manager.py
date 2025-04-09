"""
Reference model management for the ShadowCoach system.

This module handles storage and management of reference boxing techniques:
- Reference technique storage
- Model serialization/deserialization
- Technique metadata management
- Reference comparison utilities

Key Features:
    - In-memory reference storage
    - File-based persistence
    - Metadata support
    - Reference validation

Technical Details:
    - Uses pickle for model storage
    - Supports technique metadata
    - Handles versioning
    - Provides validation

Example:
    >>> manager = ReferenceManager()
    >>> manager.save_reference("pro_jab", poses, metadata)
    >>> manager.save_references_to_file("models/references.pkl")
"""
import os
from typing import Dict, Any, List, Optional

from ..utils.logging_utils import logger, timed
from ..utils.file_utils import save_pickle, load_pickle

class ReferenceManager:
    """
    Class for managing reference models for boxing techniques
    """
    def __init__(self):
        """Initialize the reference manager"""
        logger.info("Initializing ReferenceManager...")
        # Reference data storage
        self.references: Dict[str, Dict[str, Any]] = {}
        logger.info("ReferenceManager initialized successfully")

    @timed
    def save_reference(self, name: str, pose_data: List[Dict[str, Dict[str, float]]],
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a reference model

        Args:
            name: The name of the reference
            pose_data: The pose data for the reference
            metadata: Optional metadata for the reference
        """
        logger.info(f"Saving reference model: {name}")
        self.references[name] = {
            'poses': pose_data,
            'metadata': metadata or {}
        }

    @timed
    def save_references_to_file(self, path: str) -> None:
        """
        Save all references to a file

        Args:
            path: The path to save the references to
        """
        logger.info(f"Saving all references to file: {path}")
        save_pickle(self.references, path)
        logger.info(f"References saved successfully to {path}")

    @timed
    def load_references_from_file(self, path: str) -> bool:
        """
        Load references from a file

        Args:
            path: The path to load the references from

        Returns:
            True if references were loaded successfully, False otherwise
        """
        logger.info(f"Loading references from file: {path}")

        if not os.path.exists(path):
            logger.error(f"Reference file not found: {path}")
            return False

        try:
            self.references = load_pickle(path)
            logger.info(f"Loaded {len(self.references)} references successfully")

            # Log the loaded references
            for name, ref in self.references.items():
                metadata = ref.get('metadata', {})
                poses_count = len(ref.get('poses', []))
                logger.info(f"  - {name}: {poses_count} poses, metadata: {metadata}")

            return True
        except Exception as e:
            logger.error(f"Error loading references: {e}")
            return False

    def get_reference(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a reference by name

        Args:
            name: The name of the reference

        Returns:
            The reference data or None if not found
        """
        if name not in self.references:
            logger.warning(f"Reference not found: {name}")
            return None

        return self.references[name]

    def list_references(self) -> List[str]:
        """
        List all available references

        Returns:
            A list of reference names
        """
        return list(self.references.keys())
