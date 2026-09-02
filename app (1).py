import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="DustForeCast #6",
    page_icon="◌",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "21aimag_30days_cleaned.csv"
GEOJSON_PATH = Path(__file__).parent / "mongolia_aimags.geojson"
OFFICIAL_URL = "https://agaar.gov.mn"
NO_DATA_COLOR = "#c9d3cc"

AQI_LEVELS = [
    (50, "Цэвэр", "Ихэнх хүнд эрүүл мэндийн эрсдэлгүй.", "#16806a"),
    (100, "Дунд зэрэг", "Мэдрэмтгий хүмүүс удаан хугацаагаар гадаа байхаас сэргийл.", "#c08316"),
    (150, "Мэдрэмтгий бүлэгт муу", "Хүүхэд, өндөр настан, жирэмсэн хүн болон амьсгалын өвчтэй хүн анхаар.", "#d96832"),
    (200, "Эрүүл мэндэд муу", "Гадаа байх хугацааг багасгаж, маск хэрэглэ.", "#c53c4a"),
    (300, "Маш муу", "Гадаа дасгал хийхгүй, цонхоо хааж, хамгаалалт хэрэглэ.", "#8d3c72"),
    (9999, "Аюултай", "Гадагш гарахгүй байхыг зөвлөе. Эмнэлгийн зөвлөгөө шаардлагатай байж болно.", "#62233d"),
]

LEVEL_GUIDE = [
    ("Цэвэр", "AQI 0–50", "Гадаа алхах, дасгал хийхэд тохиромжтой.", "#16806a"),
    ("Дунд зэрэг", "AQI 51–100", "Ихэнх хүн хэвийн явж болно. Мэдрэмтгий хүн удаан байхаас сэргийл.", "#c08316"),
    ("Мэдрэмтгий бүлэгт муу", "AQI 101–150", "Хүүхэд, өндөр настан, жирэмсэн болон амьсгалын өвчтэй хүн анхаар.", "#d96832"),
    ("Эрүүл мэндэд муу", "AQI 151–200", "Гадаа байх хугацааг багасгаж, N95/KN95 маск хэрэглэ.", "#c53c4a"),
    ("Маш муу", "AQI 201–300", "Гадаах дасгалыг зогсоож, цонхоо хааж, шаардлагагүй бол бүү гар.", "#8d3c72"),
    ("Аюултай", "AQI 301+", "Гадагш гарахгүй байж, албан ёсны сэрэмжлүүлгийг дага.", "#62233d"),
]


def load_aimag_geometry():
    """Project the real aimag boundaries (from mongolia_aimags.geojson) into a compact
    SVG coordinate space, instead of the made-up circle layout this used to be.

    Longitude is scaled by cos(mean latitude) so the country isn't horizontally
    stretched. A handful of aimags (Ulaanbaatar, Orkhon, Darkhan-Uul, Govisumber) are
    tiny enclaves too small to hold an inline label; they're detected automatically
    (the largest gap in sorted polygon areas) and get a dot + leader-line callout
    along the top margin instead.
    """
    with open(GEOJSON_PATH, encoding="utf-8") as file:
        geojson = json.load(file)

    def rings_of(geometry):
        if geometry["type"] == "Polygon":
            return geometry["coordinates"]
        return [ring for polygon in geometry["coordinates"] for ring in polygon]

    lons = [lon for feature in geojson["features"] for ring in rings_of(feature["geometry"]) for lon, _ in ring]
    lats = [lat for feature in geojson["features"] for ring in rings_of(feature["geometry"]) for _, lat in ring]
    lon_min, lat_min, lat_max = min(lons), min(lats), max(lats)
    k = math.cos(math.radians((lat_min + lat_max) / 2))
    scale = 100 / ((max(lons) - lon_min) * k)
    pad_x, pad_top, pad_bottom = 2.0, 16.0, 2.0

    def project(lon, lat):
        return (lon - lon_min) * k * scale + pad_x, (lat_max - lat) * scale + pad_top

    def path_for(rings):
        parts = []
        for ring in rings:
            points = [project(lon, lat) for lon, lat in ring]
            parts.append("M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in points) + " Z")
        return " ".join(parts)

    def area_of(ring):
        points = [project(lon, lat) for lon, lat in ring]
        total = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]))
        return abs(total) / 2

    aimags, areas = {}, []
    for feature in geojson["features"]:
        props = feature["properties"]
        rings = rings_of(feature["geometry"])
        area = area_of(rings[0])
        areas.append(area)
        aimags[props["name_en"]] = {"path": path_for(rings), "centroid": project(props["lon"], props["lat"]), "area": area}

    sorted_areas = sorted(areas)
    gaps = [sorted_areas[i + 1] / sorted_areas[i] for i in range(len(sorted_areas) - 1)]
    split = max(range(len(gaps)), key=lambda i: gaps[i])
    small_cutoff = (sorted_areas[split] + sorted_areas[split + 1]) / 2

    map_width = (max(lons) - lon_min) * k * scale + pad_x * 2
    map_height = (lat_max - lat_min) * scale + pad_top + pad_bottom
    return aimags, small_cutoff, map_width, map_height


AIMAG_GEOMETRY, SMALL_AIMAG_CUTOFF, MAP_WIDTH, MAP_HEIGHT = load_aimag_geometry()


def load_data():
    data = pd.read_csv(DATA_PATH)
    data["DateTime"] = pd.to_datetime(data["DateTime"])
    for column in ["Temperature_C", "Humidity_percent", "Wind_m_s", "Pressure_hPa", "PM2.5_ug_m3", "PM10_ug_m3", "US_AQI"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["DateTime", "Aimag", "US_AQI"])


def aqi_info(value):
    for limit, label, advice, color in AQI_LEVELS:
        if value <= limit:
            return label, advice, color
    return AQI_LEVELS[-1][1:]


def forecast_for(data, aimag):
    history = data[data["Aimag"] == aimag].sort_values("DateTime").tail(72).copy()
    if history.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, history
    recent = history["US_AQI"].tail(24).mean()
    previous = history["US_AQI"].head(24).mean()
    trend = (recent - previous) * 0.35
    forecast = max(0.0, recent + trend)
    low = max(0.0, forecast * 0.78)
    high = forecast * 1.24 + max(2.0, history["US_AQI"].std() * 0.35)
    last = history.iloc[-1]
    return forecast, low, high, float(last["PM10_ug_m3"]), float(last["Humidity_percent"]), float(last["Wind_m_s"]), history


def future_forecast(data, aimag, target_date, weather=None):
    base, _, _, pm10, humidity, wind, history = forecast_for(data, aimag)
    if history.empty:
        return 0.0, 0.0, 0.0, pm10, humidity, wind, history
    latest_date = data["DateTime"].max().date()
    days_ahead = max(1, (target_date - latest_date).days)
    day_shift = (history["US_AQI"].tail(24).mean() - history["US_AQI"].head(24).mean()) * 0.18
    forecast = max(0.0, base + day_shift * min(days_ahead, 7))
    if weather:
        averages = history[["Temperature_C", "Humidity_percent", "Wind_m_s", "Pressure_hPa"]].mean()
        weather_effect = (
            (weather["temperature"] - averages["Temperature_C"]) * 0.35
            + (averages["Humidity_percent"] - weather["humidity"]) * 0.22
            + (averages["Wind_m_s"] - weather["wind"]) * 4.5
            + (averages["Pressure_hPa"] - weather["pressure"]) * 0.08
        )
        forecast = max(0.0, forecast + weather_effect)
    low = max(0.0, forecast * 0.78)
    high = forecast * 1.24 + max(2.0, history["US_AQI"].std() * 0.35)
    return forecast, low, high, pm10, humidity, wind, history


def hourly_series(data, aimag, start_time, weather):
    rows = []
    for offset in range(12):
        hour = start_time + pd.Timedelta(hours=offset)
        base = future_forecast(data, aimag, hour.date(), weather)[0]
        hourly_pattern = data[data["Aimag"] == aimag].groupby(data["DateTime"].dt.hour)["US_AQI"].mean()
        pattern = hourly_pattern.get(hour.hour, data[data["Aimag"] == aimag]["US_AQI"].mean())
        overall = data[data["Aimag"] == aimag]["US_AQI"].mean()
        value = max(0.0, base + (pattern - overall) * 0.35)
        rows.append({"Цаг": hour, "AQI": value})
    return pd.DataFrame(rows).set_index("Цаг")


def dust_storm_probability(aqi, wind, humidity, pm10):
    raw = (aqi * 0.18) + max(0, 2.5 - wind) * 10 + max(0, 35 - humidity) * 0.7 + pm10 * 0.9
    return int(np.clip(np.round(raw / 5) * 5, 0, 95))


def activity_advice(aqi, storm_probability):
    if storm_probability >= 70 or aqi > 150:
        return "Гадаа удаан хугацаагаар байх, гүйх, дугуй унах, хүүхдийн гадаах тоглоомыг хойшлуул."
    if storm_probability >= 40 or aqi > 100:
        return "Богино хугацаанд гарах бол маск зүү. Мөн цонхоо хаагаарай."
    return "Алхах, хөнгөн дасгал хийх боломжтой. Гадаа гарахын өмнө маск авч яваарай."


def future_series(data, aimag, start_date, days=7, weather=None):
    rows = []
    for offset in range(days):
        date = start_date + pd.Timedelta(days=offset)
        value = future_forecast(data, aimag, date.date(), weather)[0]
        rows.append({"Огноо": date, "AQI": value})
    return pd.DataFrame(rows).set_index("Огноо")


def advice_for(aqi, wind, humidity):
    if aqi <= 50:
        return "Гадаа алхах, дасгал хийхэд тохиромжтой. Энгийн хувцас хангалттай."
    if aqi <= 100:
        return "Гадаа удаан хугацаагаар байх бол амны хаалт авч яваарай."
    if aqi <= 150:
        return "Хүүхэд, өндөр настан болон харшилтай хүн гадаа байх хугацаагаа багасгах хэрэгтэй."
    if wind < 1.5:
        return "Салхи багатай тул тоос тогтоно. N95/KN95 маск зүүж, цонхоо хаахыг зөвлөе."
    if humidity < 30:
        return "Агаар хуурай байна. Тоос босох магадлал өндөр тул маск зүүж, ус сайн уугаарай."
    return "Гадаа байх хугацааг багасгаж, хамгаалалтын маск хэрэглээрэй."


def map_svg(summary, selected):
    small = sorted(
        (name for name, geo in AIMAG_GEOMETRY.items() if geo["area"] < SMALL_AIMAG_CUTOFF),
        key=lambda name: AIMAG_GEOMETRY[name]["centroid"][0],
    )
    callout_y = MAP_WIDTH * 0.08
    span = MAP_WIDTH * 0.55
    start_x = (MAP_WIDTH - span) / 2
    step = span / (len(small) - 1) if len(small) > 1 else 0
    callout_x = {name: start_x + step * i for i, name in enumerate(small)}

    shapes, leaders, labels = [], [], []
    for name, geo in AIMAG_GEOMETRY.items():
        cx, cy = geo["centroid"]
        value = summary.get(name)
        color = aqi_info(value)[2] if value is not None else NO_DATA_COLOR
        is_selected = name == selected
        is_small = name in callout_x
        stroke = "#ed7b45" if is_selected else "#17241f"
        stroke_width = ("0.4" if is_small else "0.6") if is_selected else "0.22"
        path_el = f'<path d="{geo["path"]}" fill="{color}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        shapes.append(
            f'<a href="?aimag={name}" aria-label="{html.escape(name)} сонгох"><g class="province-marker">{path_el}</g></a>'
            if value is not None else path_el
        )

        display_name = html.escape(name.replace("Ulaanbaatar", "УБ"))
        weight = ' font-weight="700"' if is_selected else ""
        no_data_class = " map-nodata" if value is None else ""
        if is_small:
            label_x = callout_x[name]
            leaders.append(
                f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{label_x:.2f}" y2="{callout_y + 2.4:.2f}" '
                f'class="map-leader" stroke-width="{"0.4" if is_selected else "0.18"}"/>'
            )
            dot = (
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{"1.3" if is_selected else "0.9"}" '
                f'fill="{color}" stroke="{stroke}" stroke-width="0.3"/>'
            )
            text = f'<text x="{label_x:.2f}" y="{callout_y:.2f}" text-anchor="middle" class="map-callout{no_data_class}"{weight}>{display_name}</text>'
            markup = dot + text
            labels.append(f'<a href="?aimag={name}"><g class="province-marker">{markup}</g></a>' if value is not None else markup)
        else:
            labels.append(
                f'<text x="{cx:.2f}" y="{cy:.2f}" text-anchor="middle" class="map-label{no_data_class}"{weight}>{display_name}</text>'
            )

    return f'''<svg class="mongolia-map" viewBox="0 0 {MAP_WIDTH:.2f} {MAP_HEIGHT:.2f}" role="img" aria-label="21 аймгийн тоосжилтын зураг">
    {''.join(shapes)}{''.join(leaders)}{''.join(labels)}
    </svg>'''


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Manrope:wght@700;800&display=swap');
:root { --ink:#17241f; --muted:#697770; --paper:#f6f3ea; --mint:#d9eee1; --orange:#ed7b45; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
.stApp { background:#f6f3ea; }
.stAppHeader, [data-testid="stToolbar"], footer { display:none; }
.block-container { max-width:1180px; padding:2.2rem 3rem 3.5rem; }
h1,h2,h3 { font-family:'Manrope', sans-serif; letter-spacing:0; }
.hero { padding:1rem 0 2rem; display:flex; justify-content:space-between; gap:2rem; align-items:end; border-bottom:1px solid #dce4da; }
.eyebrow { color:var(--orange); font-size:.76rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; }
.hero h1 { font-size:clamp(2.4rem,5vw,4.6rem); line-height:.98; margin:.5rem 0 1rem; max-width:720px; }
.hero p { color:var(--muted); max-width:590px; font-size:1.05rem; line-height:1.6; }
.source { color:var(--muted); font-size:.86rem; padding-bottom:.5rem; }
.source a { color:var(--ink); font-weight:700; }
.panel { background:#fffdf8; border:1px solid #dce4da; border-radius:8px; padding:1.25rem; box-shadow:0 4px 18px rgba(23,36,31,.035); }
.metric { background:var(--ink); color:var(--paper); border-radius:8px; padding:1.2rem; min-height:138px; }
.metric small { color:#b9c9bd; font-size:.8rem; }
.metric strong { display:block; font-family:'Manrope'; font-size:2.7rem; line-height:1.05; margin:.6rem 0 .25rem; }
.metric span { color:#d9eee1; font-weight:700; }
.map-shell { background:#d9eee1; border:1px solid #c5ded0; border-radius:8px; padding:1rem; min-height:390px; }
.mongolia-map { width:100%; height:auto; display:block; }
.mongolia-map text { font-size:1.9px; fill:#17241f; font-family:'DM Sans'; }
.map-callout { font-size:1.55px; }
.map-leader { stroke:#7db39b; stroke-dasharray:.5,.5; }
.map-nodata { fill:#697770; }
.province-marker { cursor:pointer; }
.province-marker:hover path { stroke:#ed7b45; stroke-width:.55; }
.province-marker:hover circle { stroke:#ed7b45; stroke-width:.5; }
.province-marker:hover text { font-weight:700; }
.legend { display:flex; gap:1rem; flex-wrap:wrap; margin-top:.8rem; color:var(--muted); font-size:.78rem; }
.dot { width:9px; height:9px; border-radius:50%; display:inline-block; margin-right:4px; }
.guide-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:.7rem; }
.guide-item { background:#fffdf8; border:1px solid #dce4da; border-radius:8px; padding:1rem; min-height:126px; }
.guide-item small { color:var(--orange); font-weight:700; display:block; margin-top:.55rem; }
.guide-item p { color:var(--muted); font-size:.86rem; line-height:1.45; margin:.45rem 0 0; }
.warning { border:1px solid #f0d6ad; border-left:4px solid var(--orange); background:#fff6e9; padding:1rem 1.1rem; border-radius:0 7px 7px 0; color:#68472e; line-height:1.5; }
.footer-note { color:var(--muted); font-size:.82rem; line-height:1.55; border-top:1px solid #dce4da; padding-top:1.2rem; }
[data-testid="stSidebar"] { background:#e4f0e5; }
@media (max-width: 700px) { .block-container { padding:1.5rem 1rem 3rem; } .hero { display:block; } .source { margin-top:1.5rem; } .guide-grid { grid-template-columns:1fr; } }
</style>
""", unsafe_allow_html=True)

try:
    data = load_data()
except Exception as error:
    st.error(f"Өгөгдөл уншихад алдаа гарлаа: {error}")
    st.stop()

aimag_options = sorted(data["Aimag"].unique())
first_future_date = pd.to_datetime(data["Date"].max()).date() + pd.Timedelta(days=1)
if "selected_aimag" not in st.session_state:
    st.session_state.selected_aimag = "Ulaanbaatar" if "Ulaanbaatar" in aimag_options else aimag_options[0]
clicked_aimag = st.query_params.get("aimag")
if clicked_aimag in aimag_options and clicked_aimag != st.session_state.selected_aimag:
    st.session_state.selected_aimag = clicked_aimag
    st.query_params.pop("aimag", None)
if "selected_date" not in st.session_state:
    st.session_state.selected_date = first_future_date
if st.session_state.selected_aimag not in aimag_options:
    st.session_state.selected_aimag = aimag_options[0]

with st.popover("Шүүлтүүр"):
    st.markdown("### Шүүлтүүр")
    selected_aimag = st.selectbox("Аймаг сонгох", aimag_options, key="selected_aimag")
    selected_date = st.date_input("Ирээдүйн өдөр", value=first_future_date, min_value=first_future_date, max_value=first_future_date + pd.Timedelta(days=13), key="selected_date")
    latest = data[data["Aimag"] == selected_aimag].sort_values("DateTime").iloc[-1]
    st.markdown("### Өөрийн таамаглал")
    input_temperature = st.number_input("Температур (°C)", value=float(latest["Temperature_C"]), step=0.5)
    input_humidity = st.number_input("Агаарын чийгшил (%)", min_value=0.0, max_value=100.0, value=float(latest["Humidity_percent"]), step=1.0)
    input_pressure = st.number_input("Агаарын даралт (hPa)", value=float(latest["Pressure_hPa"]), step=0.5)
    input_wind = st.number_input("Салхины хурд (м/с)", min_value=0.0, value=float(latest["Wind_m_s"]), step=0.1)
    st.markdown("---")
    st.caption("Өгөгдөл: 21 аймаг, цаг тутмын ажиглалт")
    st.caption(f"Шинэчлэгдсэн: {data['DateTime'].max():%Y-%m-%d %H:%M}")

weather_input = {
    "temperature": input_temperature,
    "humidity": input_humidity,
    "pressure": input_pressure,
    "wind": input_wind,
}
forecast, low, high, pm10, humidity, wind, history = future_forecast(data, selected_aimag, selected_date, weather_input)
label, level_advice, color = aqi_info(forecast)
storm_probability = dust_storm_probability(forecast, input_wind, input_humidity, pm10)
summary = {name: future_forecast(data, name, selected_date, weather_input if name == selected_aimag else None)[0] for name in data["Aimag"].unique()}

st.markdown(f'''<div class="hero"><div><div class="eyebrow">DustForeCast #6 · ирээдүйн төлөв · {selected_date:%Y.%m.%d}</div><h1>DustForeCast<br>#6</h1><p>21 аймгийн ажиглалтын өгөгдөлд тулгуурлан ирээдүйн өдрүүдийн тоосжилтыг таамаглаж, тухайн өдөр хэрхэн бэлдэхийг зөвлөнө.</p></div><div class="source">Албан эх сурвалж<br><a href="{OFFICIAL_URL}" target="_blank">agaar.gov.mn ↗</a></div></div>''', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(f'<div class="metric"><small>{selected_aimag} · таамагласан AQI</small><strong>{forecast:.0f}</strong><span>{label}</span></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric"><small>Тоосжилтын боломжит завсар</small><strong>{low:.0f}–{high:.0f}</strong><span>AQI</span></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric"><small>PM10 · сүүлийн ажиглалт</small><strong>{pm10:.1f}</strong><span>µg/m³</span></div>', unsafe_allow_html=True)
with col4: st.markdown(f'<div class="metric"><small>агаарын чийгшил</small><strong>{humidity:.0f}%</strong><span>салхи {wind:.1f} м/с</span></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.15, .85], gap="large")
with left:
    st.markdown("### 21 аймгийн зураг")
    st.markdown(f'<div class="map-shell">{map_svg(summary, selected_aimag)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="legend"><span><i class="dot" style="background:#16806a"></i>Цэвэр</span><span><i class="dot" style="background:#c08316"></i>Дунд зэрэг</span><span><i class="dot" style="background:#d96832"></i>Мэдрэмтгий бүлэгт муу</span><span><i class="dot" style="background:#c53c4a"></i>Эрүүл мэндэд муу</span></div>', unsafe_allow_html=True)
with right:
    st.markdown(f"### {selected_aimag} · юу хийх вэ?")
    st.markdown(f'<div class="panel"><div class="eyebrow" style="color:{color}">{label} · шороон шуурганы магадлал {storm_probability}%</div><h3 style="margin:.45rem 0">{advice_for(forecast, input_wind, input_humidity)}</h3><p style="color:#697770;line-height:1.55">{level_advice}</p><p style="color:#697770;line-height:1.55"><b>Тухайн өдөр:</b> {activity_advice(forecast, storm_probability)}</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("#### Тооцоонд ашигласан хүчин зүйлс")
    st.markdown(
        f'<div class="panel"><b>Температур:</b> {input_temperature:.1f} °C &nbsp; · &nbsp; '
        f'<b>Чийгшил:</b> {input_humidity:.1f}%<br>'
        f'<b>Даралт:</b> {input_pressure:.1f} hPa &nbsp; · &nbsp; '
        f'<b>Салхи:</b> {input_wind:.1f} м/с<br>'
        f'<b>PM10:</b> {pm10:.1f} µg/m³ &nbsp; · &nbsp; <b>Таамаг AQI:</b> {forecast:.0f}</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("#### Дараагийн 12 цагийн төлөв")
    chart = hourly_series(data, selected_aimag, pd.Timestamp(selected_date), weather_input)
    st.line_chart(chart, height=190, color="#ed7b45")

st.write("")
st.markdown("### Тоосжилтын түвшин гэж юу вэ?")
guide_html = "<div class=\"guide-grid\">"
for guide_label, range_text, description, guide_color in LEVEL_GUIDE:
    guide_html += f'<div class="guide-item"><div><i class="dot" style="background:{guide_color}"></i><b>{guide_label}</b></div><small>{range_text}</small><p>{description}</p></div>'
guide_html += "</div>"
st.markdown(guide_html, unsafe_allow_html=True)

st.write("")
st.markdown("### Тоосжилт яагаад асуудал вэ?")
explain, facts = st.columns([1.2, .8], gap="large")
with explain:
    st.markdown('<div class="panel"><p style="font-size:1.05rem;line-height:1.7;margin:0">Тоосжилт гэдэг нь агаарт хөвж байгаа маш жижиг ширхэглэг тоос юм. PM2.5 шиг жижиг хэсгүүд хамар, хоолойг цочроохоос гадна уушгинд гүн нэвтэрч болно. Тиймээс тоосжилт их өдөр маск зүүх, цонхоо хаах, гадаах идэвхтэй хөдөлгөөнийг багасгах нь хэрэгтэй.</p></div>', unsafe_allow_html=True)
with facts:
    st.markdown('<div class="warning"><b>Баримтад тулгуурласан санамж</b><br>ДЭМБ-ын зөвлөмжөөр PM2.5-ийн 24 цагийн чиглүүлэх утга 15 µg/m³ байдаг. Энэ dashboard нь AQI ба тухайн dataset-ийн чиг хандлагыг ашигладаг тул эмнэлгийн онош биш.</div>', unsafe_allow_html=True)

st.write("")
st.markdown(f'<div class="footer-note"><b>Анхааруулга:</b> Энд гарч буй тоо нь 100% үнэн баталгаа биш, өгөгдөлд суурилсан таамаглал юм. Бодит нөхцөл салхи, замын хөдөлгөөн, барилгын ажил, гал түймэр зэрэг хүчин зүйлээс шалтгаалан өөр байж болно. Албан ёсны шинэ мэдээг <a href="{OFFICIAL_URL}" target="_blank">agaar.gov.mn</a>-ээс шалгана уу.</div>', unsafe_allow_html=True)
