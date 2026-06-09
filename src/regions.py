# src/regions.py — Waiting zone rectangles for the 4 intersection approaches.
# Format: (x1, y1, x2, y2)  where (x1,y1) is top-left, (x2,y2) is bottom-right.
# A vehicle is counted in a zone only if its bounding box center falls inside it.
# To adjust zones visually, run: python src/zone_calibrator.py

N_WAIT = (727,   17,  971,  236)   # North approach
S_WAIT = (877,  897, 1172, 1076)   # South approach
E_WAIT = (1259, 326, 1914,  613)   # East approach
W_WAIT = (9,    512,  640,  815)   # West approach

WAITING_ZONES = {
    "N_WAIT": N_WAIT,
    "S_WAIT": S_WAIT,
    "E_WAIT": E_WAIT,
    "W_WAIT": W_WAIT,
}

# BGR colors for on-screen zone rendering
ZONE_COLORS = {
    "N_WAIT": (255, 100,   0),
    "S_WAIT": (0,   200, 255),
    "E_WAIT": (0,   255, 100),
    "W_WAIT": (180,   0, 255),
}
