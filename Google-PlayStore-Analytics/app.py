"""
app.py  —  Unveiling the Android App Market
=============================================
Streamlit dashboard for Google Play Store data analysis.

Run:  streamlit run app.py
"""

import os, io, warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.data_cleaning    import clean_play_store, data_quality_report
from utils.sentiment_analysis import load_reviews, sentiment_summary, top_words, wordcloud_text
from utils import visualizations as viz

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    HAS_WC = True
except ImportError:
    HAS_WC = False

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Android App Market Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.kpi-card{background:linear-gradient(135deg,#1e3a5f 0%,#2563eb 100%);border-radius:12px;
  padding:18px 22px;margin:4px 0;color:white;text-align:center;
  box-shadow:0 4px 12px rgba(0,0,0,.25);}
.kpi-value{font-size:2rem;font-weight:700;margin:6px 0;}
.kpi-label{font-size:.85rem;opacity:.85;text-transform:uppercase;letter-spacing:1px;}
.section-title{font-size:1.5rem;font-weight:700;border-left:4px solid #2563eb;
  padding-left:12px;margin:20px 0 8px;}
.insight-box{background:#f0f7ff;border-left:4px solid #2563eb;border-radius:6px;
  padding:14px 18px;margin:8px 0;}
</style>""", unsafe_allow_html=True)

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "datasets")

# ─────────────────────────────────────────────────────────
# FILE UPLOAD SECTION
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.subheader("📂 Upload Datasets")

    uploaded_apps = st.file_uploader(
        "Upload Play Store Dataset",
        type=["csv"],
        help="Upload googleplaystore.csv"
    )

    uploaded_reviews = st.file_uploader(
        "Upload Reviews Dataset (Optional)",
        type=["csv"],
        help="Upload googleplaystore_user_reviews.csv"
    )


@st.cache_data(show_spinner="Loading & cleaning data...")
def load_data(apps_file=None, reviews_file=None):

    # Main Dataset
    if apps_file is not None:
        raw = pd.read_csv(apps_file)

    else:
        apps_path = os.path.join(BASE, "googleplaystore.csv")

        if not os.path.exists(apps_path):
            st.error(
                "No dataset found.\n\n"
                "Upload a CSV file from the sidebar or place "
                "'googleplaystore.csv' inside the datasets folder."
            )
            st.stop()

        raw = pd.read_csv(apps_path)

    clean = clean_play_store(raw)
    qr = data_quality_report(raw, clean)

    # Reviews Dataset
    rev = None

    if reviews_file is not None:
        rev = load_reviews(reviews_file)

    else:
        reviews_path = os.path.join(
            BASE,
            "googleplaystore_user_reviews.csv"
        )

        if os.path.exists(reviews_path):
            rev = load_reviews(reviews_path)

    return raw, clean, rev, qr


raw_df, df, reviews_df, quality_report = load_data(
    uploaded_apps,
    uploaded_reviews
)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📱 Play Store Analysis")

    if uploaded_apps:
        st.success(f"✅ Uploaded Dataset: {uploaded_apps.name}")
    else:
        st.info("📁 Using default dataset from datasets folder")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview", "📂 Category Analysis", "⭐ Ratings Analysis",
        "📥 Installs & Popularity", "💰 Pricing Analysis",
        "📦 App Size Analysis", "💬 Sentiment Analysis",
        "💡 Insights & Recommendations",
    ])
    st.markdown("---")
    st.subheader("🔍 Filters")
    all_cats  = sorted(df['Category'].unique())
    sel_cats  = st.multiselect("Category", all_cats, default=[], placeholder="All categories")
    sel_type  = st.selectbox("App Type", ["All","Free","Paid"])
    rating_range  = st.slider("Rating Range",  1.0, 5.0, (1.0, 5.0), 0.1)
    max_installs  = int(df['Installs'].max())
    install_range = st.slider("Installs Range", 0, max_installs,
                              (0, max_installs), step=max(1, max_installs//200))
    max_price  = float(df['Price'].max())
    price_range = st.slider("Price Range ($)", 0.0, max_price, (0.0, max_price), 0.99)
    st.markdown("---")
    st.caption(f"Dataset: **{len(df):,}** apps · **{df['Category'].nunique()}** categories")

# Apply filters
filtered = df.copy()
if sel_cats:
    filtered = filtered[filtered['Category'].isin(sel_cats)]
if sel_type != "All":
    filtered = filtered[filtered['Type'] == sel_type]
filtered = filtered[
    filtered['Rating'].between(*rating_range) &
    filtered['Installs'].between(*install_range) &
    filtered['Price'].between(*price_range)
]
if len(filtered) == 0:
    st.warning("No apps match the current filters.")
    st.stop()

def kpi(label, value):
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("📱 Unveiling the Android App Market")
    st.markdown("*Comprehensive analysis of Google Play Store data — cleaned, visualised, and ready to explore.*")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi("Total Apps",     f"{len(filtered):,}")
    with c2: kpi("Categories",     filtered['Category'].nunique())
    with c3: kpi("Avg Rating",     f"{filtered['Rating'].mean():.2f} ⭐")
    with c4: kpi("Total Reviews",  f"{filtered['Reviews'].sum()/1e6:.1f}M")
    with c5: kpi("Total Installs", f"{filtered['Installs'].sum()/1e9:.2f}B")
    with c6:
        free_pct = (filtered['Type']=='Free').mean()*100
        kpi("Free Apps", f"{free_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("🧹 Data Quality Report (Before vs After Cleaning)"):
        q1,q2 = st.columns(2)
        b,a = quality_report['before'], quality_report['after']
        with q1:
            st.subheader("Before Cleaning")
            st.metric("Rows",            f"{b['rows']:,}")
            st.metric("Duplicate Apps",  f"{b['duplicates']:,}")
            st.metric("Missing Ratings", f"{b['missing_rating']:,}")
            st.metric("Total Nulls",     f"{b['nulls_total']:,}")
        with q2:
            st.subheader("After Cleaning")
            st.metric("Rows",            f"{a['rows']:,}", delta=f"{a['rows']-b['rows']}")
            st.metric("Duplicate Apps",  "0",              delta=f"-{b['duplicates']}")
            st.metric("Missing Ratings", f"{a['missing_rating']:,}")
            st.metric("Total Nulls",     f"{a['nulls_total']:,}",
                      delta=f"{a['nulls_total']-b['nulls_total']}")

    r1c1,r1c2 = st.columns(2)
    with r1c1: st.plotly_chart(viz.top_categories_bar(filtered), use_container_width=True)
    with r1c2: st.plotly_chart(viz.free_vs_paid_pie(filtered),   use_container_width=True)
    st.plotly_chart(viz.rating_histogram(filtered), use_container_width=True)

    with st.expander("🗂️ Preview Cleaned Dataset"):
        st.dataframe(filtered.head(100), use_container_width=True, hide_index=True)

    st.download_button("⬇️ Download Cleaned Dataset (CSV)",
                       filtered.to_csv(index=False).encode(),
                       "cleaned_playstore.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Category Analysis":
    st.markdown('<div class="section-title">📂 Category Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(viz.top_categories_bar(filtered, n=15), use_container_width=True)
    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(viz.avg_rating_by_category(filtered), use_container_width=True)
    with c2: st.plotly_chart(viz.installs_by_category(filtered),   use_container_width=True)
    st.markdown("#### 📋 Category Summary Table")
    cat_summary = (
        filtered.groupby('Category')
        .agg(Apps=('App','count'), Avg_Rating=('Rating','mean'),
             Total_Installs=('Installs','sum'), Total_Reviews=('Reviews','sum'))
        .reset_index().sort_values('Apps', ascending=False)
    )
    cat_summary['Avg_Rating']     = cat_summary['Avg_Rating'].round(2)
    cat_summary['Total_Installs'] = cat_summary['Total_Installs'].apply(lambda x: f"{x:,.0f}")
    cat_summary['Total_Reviews']  = cat_summary['Total_Reviews'].apply(lambda x: f"{x:,.0f}")
    st.dataframe(cat_summary, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RATINGS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⭐ Ratings Analysis":
    st.markdown('<div class="section-title">⭐ Ratings Analysis</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(viz.rating_histogram(filtered),    use_container_width=True)
    with c2: st.plotly_chart(viz.rating_box_by_type(filtered),  use_container_width=True)

    cat_ratings = filtered.groupby('Category')['Rating'].mean().sort_values()
    r1,r2 = st.columns(2)
    with r1:
        fig_low = px.bar(cat_ratings.head(10).reset_index(),
                         x='Rating', y='Category', orientation='h',
                         color='Rating', color_continuous_scale='Reds_r',
                         title='10 Lowest-Rated Categories')
        st.plotly_chart(fig_low, use_container_width=True)
    with r2:
        fig_high = px.bar(cat_ratings.tail(10).sort_values(ascending=False).reset_index(),
                          x='Rating', y='Category', orientation='h',
                          color='Rating', color_continuous_scale='Greens',
                          title='10 Highest-Rated Categories')
        st.plotly_chart(fig_high, use_container_width=True)

    st.plotly_chart(viz.rating_vs_installs(filtered), use_container_width=True)
    sample = filtered.sample(min(2000,len(filtered)), random_state=1)
    fig_rr = px.scatter(sample, x='Reviews', y='Rating', color='Category',
                        log_x=True, opacity=0.5, title='Rating vs Reviews (log scale)')
    fig_rr.update_traces(marker_size=4)
    st.plotly_chart(fig_rr, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLS & POPULARITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Installs & Popularity":
    st.markdown('<div class="section-title">📥 Installs & Popularity</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(viz.top_apps_bar(filtered,'Installs','Installs'), use_container_width=True)
    with c2: st.plotly_chart(viz.top_apps_bar(filtered,'Reviews','Reviews'),   use_container_width=True)
    st.plotly_chart(viz.installs_by_category(filtered), use_container_width=True)
    st.plotly_chart(viz.installs_vs_reviews(filtered),  use_container_width=True)
    top15cats = filtered[filtered['Category'].isin(filtered['Category'].value_counts().head(15).index)]
    if len(top15cats) > 0:
        st.plotly_chart(viz.category_heatmap(top15cats), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PRICING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Pricing Analysis":
    st.markdown('<div class="section-title">💰 Pricing Analysis</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(viz.free_vs_paid_pie(filtered),    use_container_width=True)
    with c2: st.plotly_chart(viz.price_distribution(filtered),  use_container_width=True)
    st.plotly_chart(viz.avg_price_by_category(filtered), use_container_width=True)
    st.subheader("💎 Most Expensive Apps")
    expensive = (filtered[filtered['Price']>0]
                 .nlargest(10,'Price')[['App','Category','Price','Rating','Installs']]
                 .reset_index(drop=True))
    st.dataframe(expensive, use_container_width=True, hide_index=True)
    paid = filtered[filtered['Price']>0].copy()
    if len(paid)>0:
        paid['Est_Revenue'] = paid['Price'] * paid['Installs']
        top_rev = paid.nlargest(10,'Est_Revenue')[['App','Category','Price','Installs','Est_Revenue']]
        fig_rev = px.bar(top_rev, x='Est_Revenue', y='App', orientation='h',
                         color='Est_Revenue', color_continuous_scale='Greens',
                         title='Top 10 Apps by Estimated Revenue (Price × Installs)')
        st.plotly_chart(fig_rev, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# APP SIZE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📦 App Size Analysis":
    st.markdown('<div class="section-title">📦 App Size Analysis</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(viz.size_histogram(filtered), use_container_width=True)
    with c2: st.plotly_chart(viz.size_vs_rating(filtered), use_container_width=True)
    st.plotly_chart(viz.size_box_by_category(filtered), use_container_width=True)
    s = filtered.dropna(subset=['Size_MB']).sample(min(2000,len(filtered)), random_state=9)
    fig_si = px.scatter(s, x='Size_MB', y='Installs', color='Category',
                        log_y=True, opacity=0.5, title='App Size vs Installs (log scale)')
    fig_si.update_traces(marker_size=4)
    st.plotly_chart(fig_si, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Sentiment Analysis":
    st.markdown('<div class="section-title">💬 Sentiment Analysis</div>', unsafe_allow_html=True)
    if reviews_df is None:
        st.warning("googleplaystore_user_reviews.csv not found.")
        st.stop()

    summary = sentiment_summary(reviews_df)
    k1,k2,k3 = st.columns(3)
    with k1: st.success(f"😊 Positive: **{summary['Positive']['count']:,}** ({summary['Positive']['pct']}%)")
    with k2: st.error(  f"😠 Negative: **{summary['Negative']['count']:,}** ({summary['Negative']['pct']}%)")
    with k3: st.info(   f"😐 Neutral:  **{summary['Neutral']['count']:,}** ({summary['Neutral']['pct']}%)")

    c1,c2 = st.columns(2)
    with c1: st.plotly_chart(viz.sentiment_donut(summary), use_container_width=True)
    with c2:
        fig_pol = px.histogram(reviews_df, x='Sentiment_Polarity', nbins=40,
                               color='Sentiment',
                               color_discrete_map={'Positive':'#2ECC71','Negative':'#E74C3C','Neutral':'#95A5A6'},
                               title='Polarity Distribution', barmode='overlay', opacity=0.7)
        st.plotly_chart(fig_pol, use_container_width=True)

    pos_rev = reviews_df[reviews_df['Sentiment']=='Positive']['Translated_Review']
    neg_rev = reviews_df[reviews_df['Sentiment']=='Negative']['Translated_Review']

    if HAS_WC:
        wc1,wc2 = st.columns(2)
        def _wc_fig(text, cmap, title):
            wc = WordCloud(width=700,height=350,background_color='white',
                           colormap=cmap,max_words=80).generate(text or 'nothing')
            fig,ax = plt.subplots(figsize=(7,3.5))
            ax.imshow(wc,interpolation='bilinear'); ax.axis('off')
            ax.set_title(title,fontsize=13,fontweight='bold')
            return fig
        with wc1:
            st.subheader("☁️ Positive Word Cloud")
            st.pyplot(_wc_fig(wordcloud_text(pos_rev),'Greens','Positive Words'))
        with wc2:
            st.subheader("☁️ Negative Word Cloud")
            st.pyplot(_wc_fig(wordcloud_text(neg_rev),'Reds','Negative Words'))
    else:
        st.info("Run `pip install wordcloud` to enable word clouds.")

    tw1,tw2 = st.columns(2)
    with tw1:
        st.plotly_chart(viz.top_words_bar(top_words(pos_rev,15),'😊 Top Positive Words','#2ECC71'),
                        use_container_width=True)
    with tw2:
        st.plotly_chart(viz.top_words_bar(top_words(neg_rev,15),'😠 Top Negative Words','#E74C3C'),
                        use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights & Recommendations":
    st.markdown('<div class="section-title">💡 Insights & Recommendations</div>', unsafe_allow_html=True)

    best_launch_cat = (
        filtered.groupby('Category')
        .agg(apps=('App','count'), avg_rating=('Rating','mean'),
             total_installs=('Installs','sum'))
        .assign(score=lambda x: x['avg_rating']*np.log1p(x['total_installs'])/np.log1p(x['apps']))
        .sort_values('score', ascending=False)
    )
    top_rating_cat    = filtered.groupby('Category')['Rating'].mean().idxmax()
    top_install_cat   = filtered.groupby('Category')['Installs'].sum().idxmax()
    least_compete_cat = filtered['Category'].value_counts().idxmin()
    best_launch       = best_launch_cat.index[0] if len(best_launch_cat)>0 else "N/A"
    free_avg   = filtered[filtered['Type']=='Free']['Rating'].mean()
    paid_avg   = filtered[filtered['Type']=='Paid']['Rating'].mean()
    free_inst  = filtered[filtered['Type']=='Free']['Installs'].mean()
    paid_inst  = filtered[filtered['Type']=='Paid']['Installs'].mean()

    insights = [
        ("🚀 Best Category to Launch In",
         f"**{best_launch}** scores highest on composite (rating × installs ÷ competition)."),
        ("⭐ Highest User Ratings",
         f"Apps in **{top_rating_cat}** receive the highest average user ratings."),
        ("📥 Highest Install Volume",
         f"**{top_install_cat}** dominates total installs — largest existing demand."),
        ("🌱 Least Competition",
         f"**{least_compete_cat}** has fewest apps — potentially underserved niche."),
        ("💰 Free vs Paid Performance",
         f"Free avg rating: **{free_avg:.2f}** · Paid avg rating: **{paid_avg:.2f}**  \n"
         f"Free avg installs: **{free_inst:,.0f}** · Paid avg installs: **{paid_inst:,.0f}**"),
    ]

    for title, body in insights:
        st.markdown(
            f"""
            <div style="
                background:#1e293b;
                padding:15px;
                border-radius:10px;
                margin-bottom:10px;
                border-left:5px solid #3b82f6;
                color:white;
            ">
                <h4 style="margin:0;color:#60a5fa;">{title}</h4>
                <p style="margin-top:8px;color:#e2e8f0;">
                    {body}
                </p>
            </div>
             """,
            unsafe_allow_html=True
         )

    st.markdown("#### 🏆 Top 10 Overall Apps")
    top10 = (
        filtered.assign(score=lambda x:
            x['Rating']*np.log1p(x['Installs'])*np.log1p(x['Reviews']))
        .nlargest(10,'score')[['App','Category','Rating','Installs','Reviews','Type']]
        .reset_index(drop=True)
    )
    top10.index += 1
    st.dataframe(top10, use_container_width=True)

    fig_top10 = px.bar(top10, x='Rating', y='App', orientation='h',
                       color='Category', title='Top 10 Apps — Composite Score', text='Rating')
    st.plotly_chart(fig_top10, use_container_width=True)

    report_lines = ["# Google Play Store — Insights Report\n"]
    for title, body in insights:
        report_lines.append(f"## {title}\n{body.replace('**','')}\n")
    report_lines.append("\n## Top 10 Apps\n" + top10.to_string(index=True))
    st.download_button("⬇️ Download Insights Report (.txt)",
                       '\n'.join(report_lines).encode(),
                       "insights_report.txt", "text/plain")
