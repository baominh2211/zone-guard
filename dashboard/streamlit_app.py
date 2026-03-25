"""ZoneGuard Dashboard."""
import os, httpx, streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="ZoneGuard", page_icon="🛡", layout="wide")
st.sidebar.title("🛡 ZoneGuard")
page = st.sidebar.radio("Nav", ["📹 Live", "📋 Events", "📊 Stats", "🏥 Health"])

def get(path, params=None):
    try:
        r = httpx.get(f"{API}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def post(path, data):
    try:
        r = httpx.post(f"{API}{path}", json=data, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

if page == "📹 Live":
    st.title("📹 Live Monitor")
    h = get("/health")
    if h:
        st.metric("System", h["status"].upper())
    st.image(f"{API}/api/stream/cam_01", use_container_width=True)
    st.caption("Live feed with zone overlay — only detecting inside drawn zones")

elif page == "📋 Events":
    st.title("📋 Events")
    col1, col2 = st.columns(2)
    with col1:
        ft = st.selectbox("Type", ["All", "intrusion_start", "intrusion_end"])
    with col2:
        pg = st.number_input("Page", 1, 100, 1)
    params = {"page": pg, "page_size": 20}
    if ft != "All":
        params["event_type"] = ft
    data = get("/api/events", params)
    if data and data.get("events"):
        for ev in data["events"]:
            with st.expander(f"{ev['created_at'][:19]} | {ev['event_type']} | {ev['zone_name']} | {ev['confidence']:.0%}"):
                c1, c2 = st.columns([1,2])
                with c1:
                    if ev.get("snapshot_url"):
                        st.image(f"{API}{ev['snapshot_url']}", width=400)
                with c2:
                    st.json({"id": ev["id"][:8], "camera": ev["camera_id"], "zone": ev["zone_name"],
                             "track": ev["track_id"], "confidence": ev["confidence"],
                             "occupancy": ev["occupancy_count"]})
                    fc = st.columns(3)
                    with fc[0]:
                        if st.button("✅ Correct", key=f"ok_{ev['id']}"):
                            post(f"/api/events/{ev['id']}/feedback", {"feedback": "correct"})
                            st.rerun()
                    with fc[1]:
                        if st.button("❌ False +", key=f"fp_{ev['id']}"):
                            post(f"/api/events/{ev['id']}/feedback", {"feedback": "false_positive"})
                            st.rerun()
                    with fc[2]:
                        if st.button("⚠️ Missed", key=f"ms_{ev['id']}"):
                            post(f"/api/events/{ev['id']}/feedback", {"feedback": "missed"})
                            st.rerun()
        st.caption(f"Page {data['page']} | Total: {data['total']}")
    else:
        st.info("No events. Walk into the zone!")

elif page == "📊 Stats":
    st.title("📊 Analytics")
    s = get("/api/events/stats")
    if s:
        c1,c2,c3 = st.columns(3)
        c1.metric("Events", s["total_events"])
        c2.metric("FP Rate", f"{s['false_positive_rate']:.0%}")
        c3.metric("Reviewed", sum(s.get("feedback",{}).values()))

elif page == "🏥 Health":
    st.title("🏥 System Health")
    h = get("/health")
    if h:
        st.subheader(f"Status: {h['status'].upper()}")
        for name, info in h.get("components",{}).items():
            e = "🟢" if info.get("status")=="healthy" else "🔴"
            st.markdown(f"{e} **{name}**: {info.get('status','?')}")
    st.markdown(f"[API Docs]({API}/docs)")
