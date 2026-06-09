# src/counter.py — Live vehicle occupancy counter per waiting zone.

class ZoneCounter:
    """Counts how many tracked vehicles are currently inside each waiting zone."""

    def __init__(self, zones: dict):
        # zones: dict of zone_name -> (x1, y1, x2, y2)
        self.zones = zones

    def update(self, detections: list) -> dict:
        """Check each detection's center point against all zones.
        Returns dict of zone_name -> current vehicle count."""
        current_ids = {name: set() for name in self.zones}

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            for zone_name, coords in self.zones.items():
                # Normalize coords to handle any calibrator-produced swaps
                zx1, zx2 = min(coords[0], coords[2]), max(coords[0], coords[2])
                zy1, zy2 = min(coords[1], coords[3]), max(coords[1], coords[3])

                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    current_ids[zone_name].add(track_id)
                    break  # one zone per vehicle per frame

        return {name: len(ids) for name, ids in current_ids.items()}
