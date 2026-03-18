"""Test zone checker và state machine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from zone_guard.zones.zone_checker import ZoneChecker
from zone_guard.zones.state_machine import TrackZoneState, ZoneState, ZoneTransition


def test_point_in_zone():
    # Zone hình vuông ở giữa frame
    checker = ZoneChecker([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
    
    assert checker.is_inside(0.5, 0.5) == True,   "Tâm phải ở trong"
    assert checker.is_inside(0.1, 0.1) == False,   "Góc trên trái phải ở ngoài"
    assert checker.is_inside(0.9, 0.9) == False,   "Góc dưới phải phải ở ngoài"
    print("✅ Point-in-polygon OK")


def test_foot_point():
    fx, fy = ZoneChecker.bbox_to_foot_point([100, 50, 200, 300], 640, 480)
    assert abs(fx - 150/640) < 0.01
    assert abs(fy - 300/480) < 0.01
    print("✅ Foot-point OK")


def test_state_machine_basic():
    sm = TrackZoneState(dwell_frames=3, cooldown_seconds=0)
    
    # Frame 1: vào zone → ENTERING
    r = sm.update(inside=True)
    assert sm.state == ZoneState.ENTERING
    assert r is None   # Chưa đủ dwell
    
    # Frame 2: vẫn trong → vẫn ENTERING
    r = sm.update(inside=True)
    assert r is None
    
    # Frame 3: vẫn trong → INSIDE + event!
    r = sm.update(inside=True)
    assert sm.state == ZoneState.INSIDE
    assert r == ZoneTransition.INTRUSION_START
    print("✅ State machine: enter OK")
    
    # Frame 4: ra ngoài → event end
    r = sm.update(inside=False)
    assert sm.state == ZoneState.OUTSIDE
    assert r == ZoneTransition.INTRUSION_END
    print("✅ State machine: exit OK")


def test_flickering():
    sm = TrackZoneState(dwell_frames=3, cooldown_seconds=0)
    
    sm.update(inside=True)   # ENTERING
    sm.update(inside=False)  # Reset to OUTSIDE (flickering!)
    assert sm.state == ZoneState.OUTSIDE
    print("✅ Flickering protection OK")


if __name__ == "__main__":
    test_point_in_zone()
    test_foot_point()
    test_state_machine_basic()
    test_flickering()
    print("\n🎉 All zone tests passed!")