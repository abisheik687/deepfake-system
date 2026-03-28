"""
KAVACH-AI Forensic Evidence System
Cryptographic chain-of-custody and immutable evidence logging
NO API KEYS REQUIRED - All processing is local
"""

import hashlib
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from loguru import logger

from backend.config import settings


@dataclass
class EvidenceItem:
    """Single piece of evidence"""
    evidence_id: str
    timestamp: datetime
    evidence_type: str  # detection, frame, audio, alert
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    hash: str  # SHA-256 hash of evidence data


@dataclass
class ChainOfCustody:
    """Chain-of-custody record"""
    custody_id: str
    evidence_id: str
    timestamp: datetime
    action: str  # created, accessed, exported, modified
    actor: str   # system, admin, analyst
    details: Dict[str, Any]
    previous_hash: str
    current_hash: str


class MerkleTree:
    """
    Merkle tree for immutable evidence logging.
    
    Provides cryptographic proof of evidence integrity.
    """
    
    def __init__(self):
        """Initialize Merkle tree"""
        self.leaves: List[str] = []
        self.tree: List[List[str]] = []
        self.root_hash: Optional[str] = None
        
        logger.info("Merkle tree initialized")
    
    def add_leaf(self, data: str) -> str:
        """
        Add leaf to Merkle tree.
        
        Args:
            data: Data to hash and add
        
        Returns:
            Hash of data
        """
        leaf_hash = self._hash(data)
        self.leaves.append(leaf_hash)
        
        # Rebuild tree
        self._build_tree()
        
        return leaf_hash
    
    def _hash(self, data: str) -> str:
        """SHA-256 hash"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _build_tree(self):
        """Build Merkle tree from leaves"""
        if not self.leaves:
            self.tree = []
            self.root_hash = None
            return
        
        # Start with leaves as base level
        current_level = self.leaves.copy()
        self.tree = [current_level]
        
        # Build tree bottom-up
        while len(current_level) > 1:
            next_level = []
            
            # Pair up nodes and hash
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Hash pair
                    combined = current_level[i] + current_level[i + 1]
                    parent_hash = self._hash(combined)
                else:
                    # Odd node, promote as-is
                    parent_hash = current_level[i]
                
                next_level.append(parent_hash)
            
            current_level = next_level
            self.tree.append(current_level)
        
        # Root is top of tree
        self.root_hash = current_level[0]
        
        logger.debug(f"Merkle tree rebuilt: {len(self.leaves)} leaves, root={self.root_hash[:16]}...")
    
    def get_root(self) -> Optional[str]:
        """Get Merkle root hash"""
        return self.root_hash
    
    def get_proof(self, leaf_index: int) -> List[str]:
        """
        Get Merkle proof for leaf.
        
        Args:
            leaf_index: Index of leaf
        
        Returns:
            List of hashes forming proof path
        """
        if leaf_index >= len(self.leaves):
            return []
        
        proof = []
        index = leaf_index
        
        # Traverse tree levels
        for level in self.tree[:-1]:  # Exclude root
            # Find sibling
            if index % 2 == 0:  # Left child
                sibling_index = index + 1
            else:  # Right child
                sibling_index = index - 1
            
            if sibling_index < len(level):
                proof.append(level[sibling_index])
            
            # Move to parent
            index = index // 2
        
        return proof
    
    def verify_proof(self, leaf_hash: str, proof: List[str], root: str) -> bool:
        """
        Verify Merkle proof.
        
        Args:
            leaf_hash: Hash of leaf to verify
            proof: Merkle proof path
            root: Expected root hash
        
        Returns:
            True if proof is valid
        """
        current_hash = leaf_hash
        
        for sibling_hash in proof:
            # Combine with sibling
            combined = current_hash + sibling_hash
            current_hash = self._hash(combined)
        
        return current_hash == root


class EvidenceChain:
    """
    Cryptographic evidence chain.
    
    Links evidence items with hash pointers (like blockchain).
    """
    
    def __init__(self):
        """Initialize evidence chain"""
        self.chain: List[EvidenceItem] = []
        self.custody_records: List[ChainOfCustody] = []
        self.merkle_tree = MerkleTree()
        
        logger.info("Evidence chain initialized")
    
    def add_evidence(
        self,
        evidence_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvidenceItem:
        """
        Add evidence to chain.
        
        Args:
            evidence_type: Type of evidence
            data: Evidence data
            metadata: Optional metadata
        
        Returns:
            EvidenceItem
        """
        # Generate evidence ID
        evidence_id = hashlib.sha256(
            f"{evidence_type}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Create evidence item
        evidence = EvidenceItem(
            evidence_id=evidence_id,
            timestamp=datetime.utcnow(),
            evidence_type=evidence_type,
            data=data,
            metadata=metadata or {},
            hash=""  # Will be set below
        )
        
        # Hash evidence (including previous hash for chaining)
        evidence_json = json.dumps(asdict(evidence), default=str, sort_keys=True)
        previous_hash = self.chain[-1].hash if self.chain else "0" * 64
        combined = evidence_json + previous_hash
        evidence.hash = hashlib.sha256(combined.encode()).hexdigest()
        
        # Add to chain
        self.chain.append(evidence)
        
        # Add to Merkle tree
        self.merkle_tree.add_leaf(evidence.hash)
        
        # Create custody record
        self._record_custody(
            evidence_id=evidence_id,
            action="created",
            actor="system",
            details={"source": "detection_pipeline"}
        )
        
        logger.info(f"Evidence {evidence_id} added to chain (type: {evidence_type})")
        
        return evidence
    
    def _record_custody(
        self,
        evidence_id: str,
        action: str,
        actor: str,
        details: Dict[str, Any]
    ):
        """Record chain-of-custody action"""
        custody_id = hashlib.sha256(
            f"{evidence_id}_{action}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        previous_hash = self.custody_records[-1].current_hash if self.custody_records else "0" * 64
        
        record = ChainOfCustody(
            custody_id=custody_id,
            evidence_id=evidence_id,
            timestamp=datetime.utcnow(),
            action=action,
            actor=actor,
            details=details,
            previous_hash=previous_hash,
            current_hash=""  # Will be set below
        )
        
        # Hash custody record
        record_json = json.dumps(asdict(record), default=str, sort_keys=True)
        record.current_hash = hashlib.sha256(record_json.encode()).hexdigest()
        
        self.custody_records.append(record)
        
        logger.debug(f"Custody record {custody_id}: {action} by {actor}")
    
    def verify_integrity(self) -> bool:
        """
        Verify chain integrity.
        
        Returns:
            True if chain is intact
        """
        # Verify each link
        for i, evidence in enumerate(self.chain):
            # Recalculate hash
            evidence_copy = asdict(evidence)
            evidence_copy['hash'] = ""
            evidence_json = json.dumps(evidence_copy, default=str, sort_keys=True)
            previous_hash = self.chain[i-1].hash if i > 0 else "0" * 64
            combined = evidence_json + previous_hash
            expected_hash = hashlib.sha256(combined.encode()).hexdigest()
            
            if expected_hash != evidence.hash:
                logger.error(f"Chain integrity violated at evidence {evidence.evidence_id}")
                return False
        
        logger.info("Chain integrity verified ✓")
        return True
    
    def get_evidence_by_id(self, evidence_id: str) -> Optional[EvidenceItem]:
        """Get evidence by ID"""
        for evidence in self.chain:
            if evidence.evidence_id == evidence_id:
                return evidence
        return None
    
    def get_custody_history(self, evidence_id: str) -> List[ChainOfCustody]:
        """Get custody history for evidence"""
        return [r for r in self.custody_records if r.evidence_id == evidence_id]


class ForensicExporter:
    """
    Export evidence packages in multiple formats.
    
    Formats:
    - JSON: Standard structured data
    - CEF: Common Event Format (SIEM integration)
    - STIX: Structured Threat Information eXpression
    - PDF: Human-readable report
    """
    
    def __init__(self):
        """Initialize exporter"""
        logger.info("Forensic exporter initialized")
    
    def export_json(self, evidence_chain: EvidenceChain, output_path: Path) -> bool:
        """
        Export evidence chain as JSON.
        
        Args:
            evidence_chain: Evidence chain
            output_path: Output file path
        
        Returns:
            True if successful
        """
        try:
            package = {
                "metadata": {
                    "export_timestamp": datetime.utcnow().isoformat(),
                    "num_evidence_items": len(evidence_chain.chain),
                    "merkle_root": evidence_chain.merkle_tree.get_root(),
                    "chain_intact": evidence_chain.verify_integrity()
                },
                "evidence": [asdict(e) for e in evidence_chain.chain],
                "custody": [asdict(c) for c in evidence_chain.custody_records]
            }
            
            with open(output_path, 'w') as f:
                json.dump(package, f, indent=2, default=str)
            
            logger.info(f"Evidence exported to JSON: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting JSON: {e}")
            return False
    
    def export_cef(self, evidence_chain: EvidenceChain, output_path: Path) -> bool:
        """
        Export as CEF (Common Event Format) for SIEM.
        
        Args:
            evidence_chain: Evidence chain
            output_path: Output file path
        
        Returns:
            True if successful
        """
        try:
            with open(output_path, 'w') as f:
                for evidence in evidence_chain.chain:
                    # CEF format:
                    # CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
                    
                    severity = 5  # Medium (0-10 scale)
                    if evidence.evidence_type == "alert":
                        severity = 8
                    
                    cef_line = (
                        f"CEF:0|KAVACH-AI|Deepfake Detector|1.0|"
                        f"{evidence.evidence_type}|{evidence.evidence_type.title()} Detected|"
                        f"{severity}|"
                        f"src={evidence.metadata.get('stream_id', 'unknown')} "
                        f"externalId={evidence.evidence_id} "
                        f"hash={evidence.hash} "
                        f"confidence={evidence.data.get('confidence', 0)}\n"
                    )
                    
                    f.write(cef_line)
            
            logger.info(f"Evidence exported to CEF: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting CEF: {e}")
            return False
    
    def export_stix(self, evidence_chain: EvidenceChain, output_path: Path) -> bool:
        """
        Export as STIX (Structured Threat Information eXpression).
        
        Args:
            evidence_chain: Evidence chain
            output_path: Output file path
        
        Returns:
            True if successful
        """
        try:
            # Simplified STIX bundle
            stix_bundle = {
                "type": "bundle",
                "id": f"bundle--{hashlib.sha256(str(datetime.utcnow()).encode()).hexdigest()[:32]}",
                "spec_version": "2.1",
                "objects": []
            }
            
            # Add each evidence item as indicator
            for evidence in evidence_chain.chain:
                indicator = {
                    "type": "indicator",
                    "id": f"indicator--{evidence.evidence_id}",
                    "created": evidence.timestamp.isoformat(),
                    "modified": evidence.timestamp.isoformat(),
                    "name": f"Deepfake Detection: {evidence.evidence_type}",
                    "pattern": f"[deepfake:confidence > {evidence.data.get('confidence', 0)}]",
                    "pattern_type": "stix",
                    "valid_from": evidence.timestamp.isoformat(),
                    "labels": ["deepfake", evidence.evidence_type]
                }
                stix_bundle["objects"].append(indicator)
            
            with open(output_path, 'w') as f:
                json.dump(stix_bundle, f, indent=2)
            
            logger.info(f"Evidence exported to STIX: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting STIX: {e}")
            return False


# Global evidence chain
_evidence_chain: Optional[EvidenceChain] = None


def get_evidence_chain() -> EvidenceChain:
    """Get global evidence chain instance"""
    global _evidence_chain
    if _evidence_chain is None:
        _evidence_chain = EvidenceChain()
    return _evidence_chain
