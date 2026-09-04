"""State Diff Engine.

Compares the frozen environment state (before vs after execution) 
to detect silent mutations and ensure invariants hold.
"""

from typing import Any
import logging

logger = logging.getLogger(__name__)


class StateDiffEngine:
    
    def __init__(self):
        pass
        
    def compare(self, state_before: dict[str, Any], state_after: dict[str, Any], allowed_mutations: list[str]) -> list[str]:
        """Compare two states and return a list of unexpected mutations.
        
        allowed_mutations: List of object keys or paths that are allowed to change.
        Returns empty list if all mutations are allowed.
        """
        violations = []
        
        # Extremely basic diffing for Phase 1. 
        # In a real system, use deepdiff or a structural comparison library.
        for collection_name, old_collection in state_before.items():
            new_collection = state_after.get(collection_name, {})
            
            for key, old_obj in old_collection.items():
                if key not in new_collection:
                    if f"{collection_name}.deleted" not in allowed_mutations:
                        violations.append(f"Unexpected deletion: {collection_name}.{key}")
                    continue
                    
                new_obj = new_collection[key]
                for field, old_val in old_obj.items():
                    if field in new_obj and new_obj[field] != old_val:
                        mutation_path = f"{collection_name}.{field}"
                        if mutation_path not in allowed_mutations:
                            violations.append(f"Unexpected mutation: {mutation_path} changed from {old_val} to {new_obj[field]}")
                            
            for key in new_collection:
                if key not in old_collection:
                    if f"{collection_name}.created" not in allowed_mutations:
                        violations.append(f"Unexpected creation: {collection_name}.{key}")
                        
        return violations
