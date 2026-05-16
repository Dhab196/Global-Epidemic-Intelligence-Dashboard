import re
from io import BytesIO
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from bs4 import BeautifulSoup

# =========================================================
# Page setup
# =========================================================
st.set_page_config(
    page_title="Global Epidemic Intelligence Dashboard",
    page_icon="🌍",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(15,23,42,0.98) 0%, rgba(2,6,23,0.98) 100%);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 18px;
            padding: 14px 16px;
        }
        .dashboard-card {
            background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 12px;
        }
        .small-note {
            color: #94a3b8;
            font-size: 0.92rem;
        }
        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Constants
# =========================================================
SOURCE_URLS = {
    "covid_page": "https://data.who.int/dashboards/covid19/data",
    "covid_weekly": "https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-data.csv",
    "covid_table": "https://srhdpeuwpubsa.blob.core.windows.net/whdh/COVID/WHO-COVID-19-global-table-data.csv",
    "measles_page": "https://immunizationdata.who.int/global?topic=Provisional-measles-and-rubella-data&location=",
    "measles_monthly": "https://immunizationdata.who.int/docs/librariesprovider21/measles-and-rubella/404-table-web-epi-curve-data.xlsx?sfvrsn=5922ebf7_16",
    "cholera_page": "https://www.ecdc.europa.eu/en/all-topics-z/cholera/surveillance-and-disease-data/cholera-monthly",
    "cholera_map_publication": "https://www.ecdc.europa.eu/en/publications-data/geographical-distribution-cholera-cases-reported-worldwide-march-2025-march-2026",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

MAP_FIXES = {
    "United States of America": "United States",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Republic of Korea": "South Korea",
    "Czechia": "Czech Republic",
    "Myanmar/Burma": "Myanmar",
    "United Republic of Tanzania": "Tanzania",
    "Democratic Republic of the Congo": "Democratic Republic of Congo",
}

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

# =========================================================
# Helpers
# =========================================================
def fmt_int(value):
    if pd.isna(value):
        return "0"
    return f"{int(round(float(value))):,}"

def parse_number(value):
    if value is None:
        return 0
    text = str(value).strip().lower().replace(",", "").replace(".", "")
    if not text:
        return 0
    text = text.replace(" and ", " ")
    compact = text.replace(" ", "")
    if compact.isdigit():
        return int(compact)

    total = 0
    for token in re.split(r"[\s-]+", text):
        if token in NUMBER_WORDS:
            total += NUMBER_WORDS[token]
    return total

def extract_number(pattern, text):
    match = re.search(pattern, text, flags=re.I)
    return parse_number(match.group(1)) if match else 0

def extract_text(pattern, text):
    match = re.search(pattern, text, flags=re.I)
    return match.group(1).strip() if match else None

def clean_country_for_map(name):
    return MAP_FIXES.get(name, name)

def make_horizontal_bar(df, value_col, country_col, title, color_scale="Tealgrn", top_n=15):
    chart_df = df.sort_values(value_col, ascending=True).tail(top_n).copy()
    fig = px.bar(
        chart_df,
        x=value_col,
        y=country_col,
        orientation="h",
        text=value_col,
        color=value_col,
        color_continuous_scale=color_scale,
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(
        title=title,
        height=560,
        margin=dict(l=10, r=10, t=60, b=10),
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Cases / Counts",
        yaxis_title="",
    )
    return fig

def make_world_map(df, value_col, title):
    map_df = df.copy()
    map_df["Map Country"] = map_df["Country"].apply(clean_country_for_map)
    fig = px.choropleth(
        map_df,
        locations="Map Country",
        locationmode="country names",
        color=value_col,
        hover_name="Country",
        color_continuous_scale="YlOrRd",
        title=title,
    )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="gray"),
    )
    return fig

def download_csv_button(df, label, file_name):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )

# =========================================================
# Data loaders
# =========================================================
@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def load_covid_data():
    weekly = pd.read_csv(SOURCE_URLS["covid_weekly"])
    weekly["Date_reported"] = pd.to_datetime(weekly["Date_reported"], errors="coerce")

    for col in ["New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"]:
        weekly[col] = pd.to_numeric(weekly[col], errors="coerce").fillna(0)

    latest_date = weekly["Date_reported"].max()

    latest_country_snapshot = (
        weekly.sort_values("Date_reported")
        .groupby("Country", as_index=False)
        .tail(1)[["Country", "WHO_region", "Cumulative_cases", "Cumulative_deaths"]]
    )

    cutoff_28 = latest_date - pd.Timedelta(days=28)
    recent_28d = (
        weekly[weekly["Date_reported"] > cutoff_28]
        .groupby("Country", as_index=False)[["New_cases", "New_deaths"]]
        .sum()
        .rename(columns={"New_cases": "Cases_window", "New_deaths": "Deaths_window"})
    )

    summary = recent_28d.merge(latest_country_snapshot, on="Country", how="left")
    summary["Metric_window"] = "Last 28 days"
    summary["As_of_date"] = latest_date
    summary = summary.sort_values("Cases_window", ascending=False).reset_index(drop=True)

    table = pd.read_csv(SOURCE_URLS["covid_table"])
    numeric_cols = [
        "Cases - cumulative total",
        "Cases - cumulative total per 100000 population",
        "Cases - newly reported in last 7 days",
        "Cases - newly reported in last 7 days per 100000 population",
        "Cases - newly reported in last 24 hours",
        "Deaths - cumulative total",
        "Deaths - cumulative total per 100000 population",
        "Deaths - newly reported in last 7 days",
        "Deaths - newly reported in last 7 days per 100000 population",
        "Deaths - newly reported in last 24 hours",
    ]
    for col in numeric_cols:
        table[col] = pd.to_numeric(table[col], errors="coerce")

    global_row = table[table["Name"] == "Global"].iloc[0]

    return {
        "weekly": weekly,
        "summary": summary,
        "table": table,
        "latest_date": latest_date,
        "global_row": global_row,
    }

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_measles_data():
    response = requests.get(SOURCE_URLS["measles_monthly"], headers=HEADERS, timeout=60)
    response.raise_for_status()

    df = pd.read_excel(BytesIO(response.content), sheet_name="WEB")
    df = df.rename(
        columns={
            "Region": "WHO_region",
            "Country": "Country",
            "ISO3": "ISO3",
            "Year": "Year",
            "Month": "Month",
            "Measles \nsuspect": "Measles_suspect",
            "Measles \nclinical": "Measles_clinical",
            "Measles \nepi-linked": "Measles_epi_linked",
            "Measles \nlab-confirmed": "Measles_lab_confirmed",
            "Measles \ntotal": "Measles_total",
        }
    )

    for col in ["Measles_suspect", "Measles_clinical", "Measles_epi_linked", "Measles_lab_confirmed", "Measles_total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Date"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=1),
        errors="coerce"
    )

    latest_date = df["Date"].max()
    cutoff_12m = latest_date - pd.DateOffset(months=12)

    summary = (
        df[df["Date"] > cutoff_12m]
        .groupby(["Country", "ISO3", "WHO_region"], as_index=False)[
            ["Measles_total", "Measles_suspect", "Measles_lab_confirmed"]
        ]
        .sum()
        .rename(
            columns={
                "Measles_total": "Cases_window",
                "Measles_suspect": "Suspected_window",
                "Measles_lab_confirmed": "Lab_confirmed_window",
            }
        )
        .sort_values("Cases_window", ascending=False)
        .reset_index(drop=True)
    )

    summary["Metric_window"] = "Last 12 months"
    summary["As_of_date"] = latest_date

    return {
        "monthly": df,
        "summary": summary,
        "latest_date": latest_date,
    }

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_cholera_data():
    response = requests.get(SOURCE_URLS["cholera_page"], headers=HEADERS, timeout=60)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    lines = [line.strip() for line in soup.get_text("\n", strip=True).splitlines() if line.strip()]
    joined_text = re.sub(r"\s+", " ", " ".join(lines))

    global_match = re.search(
        r"Since 1 January (?P<year>\d{4}) and as of (?P<asof>[A-Za-z0-9 ]+), (?P<cases>[0-9 ,]+) cholera cases, including (?P<deaths>[0-9 ,]+) deaths, have been reported worldwide",
        joined_text,
        flags=re.I,
    )
    if not global_match:
        raise ValueError("Could not parse the global cholera summary.")

    year = int(global_match.group("year"))
    global_as_of = pd.to_datetime(global_match.group("asof"), errors="coerce")
    global_cases = parse_number(global_match.group("cases"))
    global_deaths = parse_number(global_match.group("deaths"))

    rows = []
    i = 0
    while i < len(lines) - 1:
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        detail = None
        country = None

        if line in {"South Sudan", "Zimbabwe"} and nxt == ":" and i + 2 < len(lines):
            country = line
            detail = lines[i + 2]
            i += 3
        elif line.endswith(":") and nxt.startswith("Since "):
            country = line[:-1].strip()
            detail = nxt
            i += 2
        else:
            i += 1
            continue

        if "Since 1 January" not in detail:
            continue

        rows.append(
            {
                "Country": country,
                "Cases_window": extract_number(
                    r"as of [A-Za-z0-9 ]+, ([A-Za-z0-9 ,]+) new cases", detail
                ),
                "Deaths_window": extract_number(
                    r"including ([A-Za-z0-9 ,]+) new deaths", detail
                ),
                "As_of_date": pd.to_datetime(
                    extract_text(r"Since 1 January \d{4} and as of ([A-Za-z0-9 ]+),", detail),
                    errors="coerce",
                ),
                "Cases_ytd": extract_number(
                    r"Since 1 January \d{4} and as of [A-Za-z0-9 ]+, ([A-Za-z0-9 ,]+) cases", detail
                ),
                "Deaths_ytd": extract_number(
                    r"Since 1 January \d{4} and as of [A-Za-z0-9 ]+, [A-Za-z0-9 ,]+ cases, including ([A-Za-z0-9 ,]+) deaths",
                    detail,
                ),
            }
        )

    summary = pd.DataFrame(rows).drop_duplicates(subset=["Country"]).sort_values("Cases_ytd", ascending=False)
    summary["Metric_window"] = f"Since 1 Jan {year}"

    global_summary = {
        "year": year,
        "as_of_date": global_as_of,
        "cases_ytd": global_cases,
        "deaths_ytd": global_deaths,
    }

    return {
        "summary": summary.reset_index(drop=True),
        "global": global_summary,
    }

# =========================================================
# Load data
# =========================================================
with st.spinner("Loading official surveillance data from WHO and ECDC..."):
    covid = load_covid_data()
    measles = load_measles_data()
    cholera = load_cholera_data()

# =========================================================
# Header
# =========================================================
now_utc = datetime.now(timezone.utc)

st.title("🌍 Global Epidemic Intelligence Dashboard")
st.caption(
    "Conference-grade infectious disease surveillance dashboard using official real-world data "
    "from WHO and ECDC — designed for scientific presentation, portfolio use, and job applications."
)

st.info(
    "Scientific note: disease modules below use different surveillance windows "
    "(COVID-19 = last 28 days, Measles = last 12 months, Cholera = year-to-date). "
    "These windows improve honesty and interpretability, but they should not be treated as a single pooled ranking."
)

# =========================================================
# Top metrics
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "COVID-19 | Global cumulative cases",
        fmt_int(covid["global_row"]["Cases - cumulative total"]),
        f"WHO latest weekly date: {covid['latest_date'].date()}",
    )
    st.caption(f"Global cumulative deaths: {fmt_int(covid['global_row']['Deaths - cumulative total'])}")

with col2:
    top_measles = measles["summary"].iloc[0]
    st.metric(
        "Measles | Highest 12-month country burden",
        fmt_int(top_measles["Cases_window"]),
        f"{top_measles['Country']} | latest month: {measles['latest_date'].date()}",
    )
    st.caption("Metric shown = reported measles cases in the last 12 months.")

with col3:
    st.metric(
        f"Cholera | Global YTD {cholera['global']['year']} cases",
        fmt_int(cholera["global"]["cases_ytd"]),
        f"ECDC latest update: {cholera['global']['as_of_date'].date()}",
    )
    st.caption(f"Global YTD deaths: {fmt_int(cholera['global']['deaths_ytd'])}")

st.markdown(
    f"<div class='small-note'>Dashboard generated at: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>",
    unsafe_allow_html=True,
)

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "🦠 Disease Explorer", "🌍 Country Comparison", "🧪 Methods & Sources"]
)

# =========================================================
# Tab 1 - Overview
# =========================================================
with tab1:
    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            make_horizontal_bar(
                covid["summary"],
                "Cases_window",
                "Country",
                "COVID-19 | Top countries by reported cases in the last 28 days",
                color_scale="Blues",
                top_n=15,
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            make_horizontal_bar(
                measles["summary"],
                "Cases_window",
                "Country",
                "Measles | Top countries by reported cases in the last 12 months",
                color_scale="OrRd",
                top_n=15,
            ),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            make_horizontal_bar(
                cholera["summary"],
                "Cases_ytd",
                "Country",
                f"Cholera | Top countries by YTD {cholera['global']['year']} reported cases",
                color_scale="YlOrBr",
                top_n=15,
            ),
            use_container_width=True,
        )

        source_status = pd.DataFrame(
            [
                {
                    "Dataset": "WHO COVID-19 weekly country file",
                    "Latest published date": covid["latest_date"].date(),
                    "Refresh cadence": "Weekly",
                    "Source URL": SOURCE_URLS["covid_page"],
                },
                {
                    "Dataset": "WHO provisional measles monthly file",
                    "Latest published date": measles["latest_date"].date(),
                    "Refresh cadence": "Monthly / provisional",
                    "Source URL": SOURCE_URLS["measles_page"],
                },
                {
                    "Dataset": "ECDC cholera worldwide overview",
                    "Latest published date": cholera["global"]["as_of_date"].date(),
                    "Refresh cadence": "Monthly",
                    "Source URL": SOURCE_URLS["cholera_page"],
                },
            ]
        )
        st.markdown("### Source recency table")
        st.dataframe(source_status, use_container_width=True, hide_index=True)

# =========================================================
# Tab 2 - Disease Explorer
# =========================================================
with tab2:
    disease = st.radio(
        "Choose a disease module",
        ["COVID-19", "Measles", "Cholera"],
        horizontal=True,
    )

    if disease == "COVID-19":
        st.subheader("COVID-19 surveillance")
        top_n = st.slider("Top countries to show", 5, 25, 15, key="covid_topn")

        st.plotly_chart(
            make_world_map(
                covid["summary"].head(150),
                "Cases_window",
                "COVID-19 | World map of reported cases in the last 28 days",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            make_horizontal_bar(
                covid["summary"],
                "Cases_window",
                "Country",
                "Top countries by reported cases in the last 28 days",
                color_scale="Blues",
                top_n=top_n,
            ),
            use_container_width=True,
        )

        default_countries = (
            covid["summary"].head(5)["Country"].tolist()
            if len(covid["summary"]) >= 5 else covid["summary"]["Country"].tolist()
        )

        selected_countries = st.multiselect(
            "Countries for weekly trend",
            options=sorted(covid["weekly"]["Country"].dropna().unique().tolist()),
            default=default_countries,
            key="covid_country_multiselect",
        )

        trend_df = covid["weekly"][covid["weekly"]["Country"].isin(selected_countries)].copy()
        trend_fig = px.line(
            trend_df,
            x="Date_reported",
            y="New_cases",
            color="Country",
            title="Weekly reported new COVID-19 cases",
        )
        trend_fig.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(trend_fig, use_container_width=True)

        covid_table = covid["summary"][
            ["Country", "WHO_region", "Cases_window", "Deaths_window", "Cumulative_cases", "Cumulative_deaths", "As_of_date"]
        ].copy()
        covid_table["As_of_date"] = covid_table["As_of_date"].dt.date
        st.dataframe(covid_table.head(50), use_container_width=True, hide_index=True)
        download_csv_button(covid_table, "Download COVID-19 table as CSV", "covid_surveillance_table.csv")

    elif disease == "Measles":
        st.subheader("Measles surveillance")
        top_n = st.slider("Top countries to show", 5, 25, 15, key="measles_topn")

        st.plotly_chart(
            make_world_map(
                measles["summary"].head(150),
                "Cases_window",
                "Measles | World map of reported cases in the last 12 months",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            make_horizontal_bar(
                measles["summary"],
                "Cases_window",
                "Country",
                "Top countries by reported measles cases in the last 12 months",
                color_scale="OrRd",
                top_n=top_n,
            ),
            use_container_width=True,
        )

        default_countries = (
            measles["summary"].head(5)["Country"].tolist()
            if len(measles["summary"]) >= 5 else measles["summary"]["Country"].tolist()
        )

        selected_countries = st.multiselect(
            "Countries for monthly trend",
            options=sorted(measles["monthly"]["Country"].dropna().unique().tolist()),
            default=default_countries,
            key="measles_country_multiselect",
        )

        trend_df = measles["monthly"][measles["monthly"]["Country"].isin(selected_countries)].copy()
        trend_df = trend_df[trend_df["Date"] >= (measles["latest_date"] - pd.DateOffset(months=24))]

        trend_fig = px.line(
            trend_df,
            x="Date",
            y="Measles_total",
            color="Country",
            title="Monthly reported measles cases (last 24 months shown)",
        )
        trend_fig.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(trend_fig, use_container_width=True)

        measles_table = measles["summary"][
            ["Country", "WHO_region", "Cases_window", "Suspected_window", "Lab_confirmed_window", "As_of_date"]
        ].copy()
        measles_table["As_of_date"] = measles_table["As_of_date"].dt.date
        st.dataframe(measles_table.head(50), use_container_width=True, hide_index=True)
        download_csv_button(measles_table, "Download Measles table as CSV", "measles_surveillance_table.csv")

    else:
        st.subheader("Cholera surveillance")
        top_n = st.slider("Top countries to show", 5, 25, 15, key="cholera_topn")

        st.plotly_chart(
            make_world_map(
                cholera["summary"],
                "Cases_ytd",
                f"Cholera | World map of YTD {cholera['global']['year']} reported cases",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            make_horizontal_bar(
                cholera["summary"],
                "Cases_ytd",
                "Country",
                f"Top countries by cholera cases since 1 Jan {cholera['global']['year']}",
                color_scale="YlOrBr",
                top_n=top_n,
            ),
            use_container_width=True,
        )

        deaths_fig = px.bar(
            cholera["summary"].sort_values("Deaths_ytd", ascending=False).head(top_n),
            x="Country",
            y="Deaths_ytd",
            color="Deaths_ytd",
            color_continuous_scale="Reds",
            title=f"Cholera | YTD {cholera['global']['year']} reported deaths by country",
            text="Deaths_ytd",
        )
        deaths_fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        deaths_fig.update_layout(height=500, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(deaths_fig, use_container_width=True)

        cholera_table = cholera["summary"][
            ["Country", "Cases_window", "Deaths_window", "Cases_ytd", "Deaths_ytd", "As_of_date"]
        ].copy()
        cholera_table["As_of_date"] = pd.to_datetime(cholera_table["As_of_date"], errors="coerce").dt.date
        st.dataframe(cholera_table, use_container_width=True, hide_index=True)
        download_csv_button(cholera_table, "Download Cholera table as CSV", "cholera_surveillance_table.csv")

# =========================================================
# Tab 3 - Country Comparison
# =========================================================
with tab3:
    comparison_disease = st.selectbox(
        "Choose the dataset to compare countries",
        ["COVID-19", "Measles", "Cholera"],
    )

    if comparison_disease == "COVID-19":
        compare_df = covid["summary"][["Country", "WHO_region", "Cases_window", "Deaths_window", "As_of_date"]].copy()
        compare_df = compare_df.rename(columns={"Cases_window": "Cases", "Deaths_window": "Deaths"})
        compare_df["Window"] = "Last 28 days"

    elif comparison_disease == "Measles":
        compare_df = measles["summary"][["Country", "WHO_region", "Cases_window", "As_of_date"]].copy()
        compare_df = compare_df.rename(columns={"Cases_window": "Cases"})
        compare_df["Deaths"] = pd.NA
        compare_df["Window"] = "Last 12 months"

    else:
        compare_df = cholera["summary"][["Country", "Cases_ytd", "Deaths_ytd", "As_of_date"]].copy()
        compare_df = compare_df.rename(columns={"Cases_ytd": "Cases", "Deaths_ytd": "Deaths"})
        compare_df["WHO_region"] = pd.NA
        compare_df["Window"] = f"Since 1 Jan {cholera['global']['year']}"

    top_n_compare = st.slider("Rows to display", 10, 100, 30, key="compare_rows")
    compare_df = compare_df.sort_values("Cases", ascending=False).head(top_n_compare)
    compare_df["As_of_date"] = pd.to_datetime(compare_df["As_of_date"], errors="coerce").dt.date

    st.dataframe(compare_df, use_container_width=True, hide_index=True)
    download_csv_button(compare_df, "Download comparison table as CSV", "country_comparison.csv")

# =========================================================
# Tab 4 - Methods & Sources
# =========================================================
with tab4:
    st.markdown("### Why this dashboard is scientifically stronger")
    st.markdown(
        """
        - It uses official surveillance sources instead of synthetic values.
        - It displays the latest available publication date from each source.
        - It separates diseases by surveillance window to avoid misleading direct comparisons.
        - It removes unsupported homemade epidemiological metrics from the core display.
        - It includes transparent source URLs for reproducibility and peer review.
        """
    )

    st.markdown("### Source catalogue")
    sources_df = pd.DataFrame(
        [
            {
                "Disease / Module": "COVID-19",
                "Primary source": "WHO COVID-19 dashboard data",
                "URL": SOURCE_URLS["covid_page"],
                "Update frequency": "Weekly",
                "Metric used in dashboard": "Reported cases in last 28 days + cumulative totals",
            },
            {
                "Disease / Module": "Measles",
                "Primary source": "WHO provisional measles and rubella monthly data",
                "URL": SOURCE_URLS["measles_page"],
                "Update frequency": "Monthly / provisional",
                "Metric used in dashboard": "Reported measles cases in last 12 months",
            },
            {
                "Disease / Module": "Cholera",
                "Primary source": "ECDC cholera worldwide overview",
                "URL": SOURCE_URLS["cholera_page"],
                "Update frequency": "Monthly",
                "Metric used in dashboard": f"Reported cholera cases since 1 Jan {cholera['global']['year']}",
            },
        ]
    )
    st.dataframe(sources_df, use_container_width=True, hide_index=True)

    st.markdown("### Recommended citation language for your project")
    st.code(
        "Global Epidemic Intelligence Dashboard. Real-world infectious disease surveillance dashboard "
        "built in Streamlit using official country-level data from WHO and ECDC. "
        "The dashboard reports disease-specific surveillance windows and source recency dates "
        "to improve transparency, reproducibility, and scientific interpretability.",
        language="text",
    )

    st.markdown("### Important caution for conference use")
    st.warning(
        "Do not claim these three diseases are directly rank-comparable across a single uniform time scale. "
        "In your talk or poster, say clearly that each module follows the latest appropriate reporting window of its source."
    )

    st.markdown("### Visual reference")
    st.markdown(
        f"You can also cite the published ECDC global cholera distribution figure as a visual reference in your abstract or poster: {SOURCE_URLS['cholera_map_publication']}"
    )

