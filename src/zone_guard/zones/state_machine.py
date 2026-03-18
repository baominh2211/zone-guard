"""State machine for track-zone interaction."""
import time
from dataclasses import dataclass, field
from enum import Enum


class ZoneState(str, Enum):
    OUTSIDE = "outside"
    ENTERING = "entering"
    INSIDE = "inside"


class ZoneTransition(str, Enum):
    INTRUSION_START = "intrusion_start"
    INTRUSION_END = "intrusion_end"


@dataclass
class TrackZoneState:
    """State machine cho 1 track trong 1 zone."""
    dwell_frames: int = 3              # Số frame liên tiếp phải ở trong zone
    cooldown_seconds: float = 60.0     # Không tạo event lại trong thời gian này

    state: ZoneState = field(default=ZoneState.OUTSIDE)
    _inside_count: int = field(default=0, repr=False)
    _last_event_time: float = field(default=0.0, repr=False)

    def update(self, inside: bool):
        """
        Gọi mỗi frame với kết quả inside=True/False.
        Trả về ZoneTransition nếu có event, None nếu không.
        """
        if self.state == ZoneState.OUTSIDE:
            if inside:
                self.state = ZoneState.ENTERING
                self._inside_count = 1
            return None

        elif self.state == ZoneState.ENTERING:
            if inside:
                self._inside_count += 1
                if self._inside_count >= self.dwell_frames:
                    # Đã ở đủ lâu → chuyển sang INSIDE
                    self.state = ZoneState.INSIDE
                    now = time.time()
                    if (now - self._last_event_time) >= self.cooldown_seconds:
                        self._last_event_time = now
                        return ZoneTransition.INTRUSION_START
            else:
                # Ra trước khi đủ dwell → reset (flickering)
                self.state = ZoneState.OUTSIDE
                self._inside_count = 0
            return None

        elif self.state == ZoneState.INSIDE:
            if not inside:
                self.state = ZoneState.OUTSIDE
                self._inside_count = 0
                return ZoneTransition.INTRUSION_END
            return None

        return None

    @property
    def is_inside(self) -> bool:
        return self.state == ZoneState.INSIDE


class TrackZoneStateManager:
    """Quản lý state machine cho tất cả cặp (track, zone)."""

    def __init__(self, dwell_frames: int = 3, cooldown_seconds: float = 60.0):
        self._states: dict[tuple[int, str], TrackZoneState] = {}
        self._dwell = dwell_frames
        self._cooldown = cooldown_seconds

    def update(self, track_id: int, zone_id: str, inside: bool):
        """Cập nhật state cho 1 track trong 1 zone. Trả về transition hoặc None."""
        key = (track_id, zone_id)
        if key not in self._states:
            self._states[key] = TrackZoneState(
                dwell_frames=self._dwell,
                cooldown_seconds=self._cooldown,
            )
        return self._states[key].update(inside)

    def get_occupancy(self, zone_id: str) -> int:
        """Đếm số track đang INSIDE zone."""
        return sum(
            1 for (_, zid), state in self._states.items()
            if zid == zone_id and state.is_inside
        )

    def cleanup_stale(self, active_track_ids: set[int]):
        """Xóa state của track đã mất (không còn trong frame)."""
        stale_keys = [k for k in self._states if k[0] not in active_track_ids]
        transitions = []
        for key in stale_keys:
            state = self._states[key]
            if state.is_inside:
                transitions.append((key[1], ZoneTransition.INTRUSION_END))
            del self._states[key]
        return transitions