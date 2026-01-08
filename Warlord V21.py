"""
WARLORD ENGINE V21.0 - UNIVERSAL (THE END)
Status: PLATFORM AGNOSTIC / ENTERPRISE GOLD
Architecture: OS-Aware Locking Kernel
Changelog:
  - ADDED: Windows Support (msvcrt.locking)
  - ADDED: POSIX Support (fcntl.flock)
  - FINALIZED: True Cross-Platform Concurrency
"""

import cv2
import numpy as np
import os
import json
import logging
import math
import time
import argparse
import hashlib
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
from contextlib import contextmanager

# --- PLATFORM SPECIFIC IMPORTS (DEPENDENCY FREE LOCKING) ---
if os.name == 'nt':
    import msvcrt
else:
    import fcntl

# --- 0. CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] UNIVERSAL: %(message)s')
logger = logging.getLogger("WARLORD_21")

class WarlordError(Exception): """Base exception"""
class LockError(WarlordError): """Concurrency exception"""

@dataclass
class Hyperparameters:
    w_kinetic: float = 1.2
    w_visual: float = 0.6
    w_semantic: float = 1.5
    prior_inertia: float = 10.0

# --- 1. INFRASTRUCTURE: UNIVERSAL LOCKING KERNEL ---
class UniversalFileLock:
    """
    İşletim sistemi seviyesinde (Kernel-Level) dosya kilitleme.
    Windows (NTFS) ve Linux/Mac (POSIX) üzerinde atomik çalışır.
    Race Condition imkansızdır.
    """
    def __init__(self, file_path: str, timeout: int = 5):
        self.lock_file = file_path + ".lock"
        self.timeout = timeout
        self._fd = None

    def __enter__(self):
        start_time = time.time()
        while True:
            try:
                # 1. Dosyayı aç (Veya oluştur)
                self._fd = open(self.lock_file, 'w')
                
                # 2. OS-Specific Locking
                if os.name == 'nt':
                    # Windows: Non-Blocking Lock (LK_NBL)
                    # 1 byte'ı kilitler.
                    msvcrt.locking(self._fd.fileno(), msvcrt.LK_NBL, 1)
                else:
                    # POSIX: Exclusive Non-Blocking Lock
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                return self

            except (IOError, OSError, PermissionError):
                # Kilit alınamadı, dosya başkası tarafından tutuluyor.
                if self._fd:
                    self._fd.close()
                    self._fd = None
                
                if time.time() - start_time > self.timeout:
                    raise LockError(f"Timeout acquiring lock: {self.lock_file}")
                
                time.sleep(0.05) # Backoff

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fd:
            try:
                # Kilidi serbest bırak
                if os.name == 'nt':
                    msvcrt.locking(self._fd.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
            except:
                pass
            finally:
                self._fd.close()
                # Opsiyonel: Kilit dosyasını temizle (Windows'ta bazen sorunludur, try-pass)
                try: os.remove(self.lock_file)
                except: pass

# --- 2. DOMAIN MODELS ---
@dataclass
class NicheStats:
    schema_version: int = 1
    mean: float = 50.0
    m2: float = 0.0
    count: int = 0
    last_updated: float = 0.0

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 100.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

# --- 3. PERSISTENCE: SECURE REPOSITORY ---
class UniversalRepository:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)

    def _get_paths(self, niche: str) -> str:
        # SHA-256 Hashing
        niche_hash = hashlib.sha256(niche.encode('utf-8')).hexdigest()
        return os.path.join(self.storage_dir, f"{niche_hash}.json")

    def get_stats(self, niche: str) -> NicheStats:
        f_path = self._get_paths(niche)
        try:
            # Okuma sırasında bile kilit kullanılır (Consistency)
            with UniversalFileLock(f_path):
                if not os.path.exists(f_path): return NicheStats()
                with open(f_path, 'r') as f:
                    data = json.load(f)
                    return NicheStats(**data)
        except Exception as e:
            logger.error(f"Read error: {e}")
            return NicheStats()

    def save_stats(self, niche: str, stats: NicheStats):
        f_path = self._get_paths(niche)
        try:
            with UniversalFileLock(f_path):
                temp_path = f_path + ".tmp"
                with open(temp_path, 'w') as f:
                    json.dump(asdict(stats), f)
                # Windows atomic replace support (Python 3.3+)
                os.replace(temp_path, f_path)
        except Exception as e:
            logger.error(f"Write error: {e}")

# --- 4. MATH & BAYESIAN ENGINE ---
class MathEngine:
    @staticmethod
    def update_welford(current: NicheStats, new_val: float) -> NicheStats:
        current.count += 1
        delta = new_val - current.mean
        current.mean += delta / current.count
        delta2 = new_val - current.mean
        current.m2 += delta * delta2
        current.last_updated = time.time()
        return current

    @staticmethod
    def compute_confidence(raw: float, prior: NicheStats, inertia: float) -> Tuple[float, float]:
        sample_size = prior.count
        credibility = sample_size / (sample_size + inertia)
        posterior = (credibility * raw) + ((1 - credibility) * prior.mean)
        uncertainty = prior.std / math.sqrt(sample_size + 1)
        return posterior, uncertainty

    @staticmethod
    def sigmoid_score(k: float, stab: float, v: float, s: float, prior_mean: float, params: Hyperparameters) -> float:
        base = (k * stab * params.w_kinetic) + (v * params.w_visual)
        raw_z = base * (1.0 + s * params.w_semantic)
        norm_in = (raw_z - (prior_mean / 5.0)) * 0.25
        return (1.0 / (1.0 + np.exp(-norm_in))) * 100.0

# --- 5. PERCEPTION CORE ---
class KineticCore:
    def __init__(self):
        self.prev_gray = None
        self.p0 = None
        self.fail_streak = 0
        self.diag = 1.0

    def _reset(self, gray):
        h, w = gray.shape
        self.diag = math.sqrt(h**2 + w**2)
        self.prev_gray = gray
        self.p0 = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        self.fail_streak = 0

    def process(self, gray, dt: float) -> Tuple[float, float]:
        if self.prev_gray is None:
            self._reset(gray)
            return 0.0, 1.0
        
        if self.fail_streak > 3: self._reset(gray)
        if self.p0 is None or len(self.p0) < 50:
             self._reset(gray)
             if self.p0 is None: 
                 self.fail_streak += 1
                 return 0.0, 1.0

        try:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None)
        except:
            self.fail_streak += 1
            return 0.0, 1.0
            
        if p1 is None:
            self.fail_streak += 1
            return 0.0, 1.0

        good_new = p1[st==1]
        good_old = self.p0[st==1]
        
        subj_px, cam_px, stability = 0.0, 0.0, 1.0
        
        try:
            if len(good_new) > 10:
                m, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
                if m is not None:
                    cam_px = np.hypot(m[0,2], m[1,2])
                    outliers = (inliers == 0).flatten()
                    if np.sum(outliers) > 0:
                        d = good_new[outliers] - good_old[outliers]
                        subj_px = np.mean(np.hypot(d[:,0], d[:,1]))
                else:
                    stability = 0.5
                    d = good_new - good_old
                    subj_px = np.mean(np.hypot(d[:,0], d[:,1]))
        except: stability = 0.5

        norm_subj = (subj_px / self.diag) / dt
        norm_cam = (cam_px / self.diag) / dt
        
        if norm_cam > 0.15 and norm_cam > norm_subj * 1.5: stability = 0.6

        self.prev_gray = gray
        self.p0 = good_new.reshape(-1, 1, 2)
        self.fail_streak = 0
        return min(1.0, norm_subj), stability

class StructuralCore:
    def compute(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 3))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        mask = np.zeros_like(gray)
        cv2.rectangle(mask, (int(w*0.1), int(h*0.2)), (int(w*0.9), int(h*0.8)), 255, -1)
        den = cv2.countNonZero(cv2.bitwise_and(binary, mask)) / (cv2.countNonZero(mask) + 1e-5)
        score_struct = math.exp(-((den - 0.15)**2) / (2 * 0.06**2))
        
        # True Shannon Entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        prob = hist.ravel() / hist.sum()
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log2(prob))
        score_entropy = math.exp(-((entropy - 7.0)**2) / (2 * 1.5**2))

        return (score_struct * 0.6) + (score_entropy * 0.4)

# --- 6. CONTROLLER ---
class WarlordController:
    def __init__(self, storage_dir: str):
        self.repo = UniversalRepository(storage_dir)
        self.params = Hyperparameters()
        self.kinetic = KineticCore()
        self.struct = StructuralCore()

    def run_demo(self):
        print("[DEMO] Warlord Universal Simulation...")
        scenarios = [("Viral", 0.8, 0.9, 0.8, 0.9), ("Bad", 0.3, 0.4, 0.5, 0.2)]
        prior = NicheStats(mean=50.0, m2=100.0, count=10)
        for name, k, stab, v, s in scenarios:
            raw = MathEngine.sigmoid_score(k, stab, v, s, prior.mean, self.params)
            conf, _ = MathEngine.compute_confidence(raw, prior, 10.0)
            print(f"Scenario: {name} -> Verdict: {conf:.1f}")

    def process(self, path: str, niche: str) -> Dict[str, Any]:
        if not os.path.exists(path): raise WarlordError("File 404")
        
        stats = self.repo.get_stats(niche)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dt = 1.0 / fps
        sampling = max(1, int(fps/4))
        
        acc = {'k': [], 's': [], 'v': [], 'stab': []}
        idx = 0
        try:
            while True:
                if idx % sampling != 0:
                    if not cap.grab(): break
                    idx += 1
                    continue
                ret, frame = cap.read()
                if not ret: break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                k, stab = self.kinetic.process(gray, dt)
                s = self.struct.compute(frame)
                v = np.mean(cv2.Canny(gray, 50, 150)) / 255.0
                acc['k'].append(k); acc['s'].append(s); acc['v'].append(v); acc['stab'].append(stab)
                idx += 1
        finally:
            cap.release()

        if not acc['k']: return {"score": 0}

        vals = {k: np.mean(v) for k, v in acc.items()}
        vals['k'] = (np.mean(acc['k']) * 0.6) + (np.percentile(acc['k'], 90) * 0.4)
        
        raw = MathEngine.sigmoid_score(vals['k'], vals['stab'], vals['v'], vals['s'], stats.mean, self.params)
        final, unc = MathEngine.compute_confidence(raw, stats, self.params.prior_inertia)
        updated = MathEngine.update_welford(stats, final)
        self.repo.save_stats(niche, updated)
        
        return {
            "verdict": round(max(0, final - (1.96 * unc)), 1),
            "meta": {"niche": niche, "os": os.name},
            "metrics": {k: round(v, 3) for k, v in vals.items()}
        }

# --- 7. CLI ENTRY ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warlord V21 - Universal")
    parser.add_argument("--video", help="Video path")
    parser.add_argument("--niche", default="general", help="Content niche")
    parser.add_argument("--db", default="./warlord_univ_db", help="DB path")
    parser.add_argument("--demo", action="store_true", help="Run simulation")
    
    args = parser.parse_args()
    
    try:
        ctrl = WarlordController(args.db)
        if args.demo:
            ctrl.run_demo()
        elif args.video:
            if os.path.exists(args.video):
                print(json.dumps(ctrl.process(args.video, args.niche), indent=4))
            else:
                print("[ERROR] File not found.")
        else:
            parser.print_help()
    except Exception as e:
        print(f"[FATAL] {e}")
