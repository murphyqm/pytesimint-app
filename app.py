import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.lines import Line2D
import streamlit as st

st.set_page_config(layout="wide", page_title="Pallasite cooling rates")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.title('Fast and slow cooling in Pallasite Meteories')


st.write("##### How much of the intrusion region should preserve calcium zoning in olivine crystals?")

cola, colb = st.columns([0.8, 0.2])
with cola:
    zoning_cons = st.slider('Zoning constraint ($\geq$ selection, vol. %)',
                            value=90, min_value=5, max_value=100,
                            label_visibility="hidden")

with colb:
    st.write(f"### $\geq$ {zoning_cons} vol. %")


st.write("##### How much of the intrusion region should facilitate macroscale rounding of olivine crystals?")

colc, cold = st.columns([0.8, 0.2])

with colc:
    rounding_cons = st.slider('Rounding constraint ($\geq$ selection, vol. %)',
                            value=70, min_value=5, max_value=100,
                            label_visibility="hidden")

with cold:
    st.write(f"### $\geq$ {rounding_cons} vol. %")

@st.cache_data
def load_dataframe(path):
    data = pd.read_csv(path)
    data_calculations = data.copy(deep=True)
    return data_calculations



data_calculations = load_dataframe("results.csv")

# initial analysis of dataframe:

@st.cache_data
def initial_analysis(data_calculations):
    data_calculations["dT01"] = ((data_calculations["mean_it1"] - data_calculations["mean_it0"])/(48.0-3.0)) * 12.0
    data_calculations["dT12"] = ((data_calculations["mean_it2"] - data_calculations["mean_it1"])/(96.0 - 48.0)) * 12.0
    data_calculations["dT23"] = ((data_calculations["mean_it3"] - data_calculations["mean_it2"])/(120.0 - 96.0)) * 12.0
    data_calculations["dT03"] = ((data_calculations["mean_it3"] - data_calculations["mean_it0"])/(96.0 - 3.0)) * 12.0

    data_calculations["r_z/r_x"] = data_calculations["r_z"]/data_calculations["r_x"]
    data_calculations["r_y/r_x"] = data_calculations["r_y"]/data_calculations["r_x"]

    data_calculations["unique/non-unique"] = data_calculations["r_z/r_x"]
    data_calculations.loc[data_calculations.orientation == "x=z", "unique/non-unique"] = data_calculations["r_y/r_x"]

    data_calculations["avg_r"] = (data_calculations["r_x"] + data_calculations["r_y"] + data_calculations["r_z"])/3

    data_calculations["non-uni"] = data_calculations["r_x"]
    data_calculations["uni"] = data_calculations["r_z"]
    data_calculations.loc[data_calculations.orientation == "x=z", "uni"] = data_calculations["r_y"]

    data_calculations["avg_mantle_temp"] = (data_calculations["temp_top"] + data_calculations["temp_bottom"])/2
    return data_calculations

# do the initial analysis
data_calculations = initial_analysis(data_calculations)


def constraint_calc(zoning_constraint, rounding_constraint, data_calculations):
    data_calculations["Zoned"] = (data_calculations.percent_geochem_preserved) >= zoning_constraint

    data_calculations["Round"] = (data_calculations.percent_rounded) >= rounding_constraint

    data_calculations['Score'] = ((data_calculations["Zoned"]==True).astype(int)
                    + (data_calculations["Round"]==True).astype(int))

    data_calculations["Criteria_Matches"] = "No match"
    data_calculations.loc[data_calculations.Score == 2, "Criteria_Matches"] = "Both match"
    data_calculations.loc[((data_calculations.Zoned == True) &
                        ((data_calculations.Score) == 1)), "Criteria_Matches"] = "Zoning matches"
    data_calculations.loc[((data_calculations.Round == True) &
                        ((data_calculations.Score) == 1)), "Criteria_Matches"] = "Rounding matches"

    data_calculations["Model_type"] = "Planetesimal model"
    data_calculations.loc[data_calculations.identifier == "random", "Model_type"] = "Randomly assigned"
    data_calculations["Model_result"] = "Does not reproduce all results"
    data_calculations.loc[(((data_calculations.Model_type) == "Planetesimal model") &
                        ((data_calculations.Score) == 2)), "Model_result"] = "Reproduces all results"
    
    passing_models = data_calculations.query("Criteria_Matches == 'Both match'").sort_values("avg_mantle_temp", ascending=False).reset_index(drop=True)
    # passing_models

    df = data_calculations.query("dT03 <= 10").sort_values("Model_result")
    # df = data_calculations.query("dT01 <= 5").query("dT12 <= 5").query("dT23 <= 5").query("dT03 <= 5").sort_values("Model_result")
    df2 = df.query("Model_result == 'Reproduces all results'")
    df_plan = df.query("identifier != 'random'")
    return passing_models, df, df2, df_plan

# run analysis with updated values
passing_models, df, df2, df_plan = constraint_calc(zoning_cons, rounding_cons, data_calculations)

# plot figure
def plotting_values(zoning_constraint,
                    rounding_constraint,
                    data_calculations,
                    passing_models,
                    df, df2, df_plan):
    annotate = ["a)", "b)", "c)",
                "d)", "e)", "f)",
                "g)", "h)", "i)",
                "j)", "k)", "l)",
                "m)", "n)", "o)", "p)"]

    pal = ['#1B98E0','#8A45C6','#e66101', '#CA054D', ]

    x_variables = ["actual_volume", "avg_mantle_temp", "int_temp",]
    y_variables = ["percent_geochem_preserved", "percent_rounded", "unique/non-unique",]

    g = sns.pairplot(df, hue="Criteria_Matches", x_vars=x_variables, y_vars=y_variables, kind="scatter",
                    hue_order=["No match", "Rounding matches", "Zoning matches", "Both match"],
                plot_kws={'alpha':0.6,
                        "style":data_calculations["Model_type"],
                        "markers":["o", "X"],
                        "size":data_calculations["Model_type"],
                        "sizes":(35, 70)},
                palette=pal,)

    for ax, title in zip(g.axes.flat, annotate):
        # ax.set_title(title)
        t = ax.text(0.04, 0.97, title,
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white',
                            alpha=0.65, edgecolor="white"))

    g.axes[0, 0].set_ylabel('Zoning preservation\n[vol. % of int. region]')
    g.axes[1, 0].set_ylabel('Rounding potential\n[vol. % of int. region]')
    g.axes[2, 0].set_ylabel("Aspect ratio\n(unique/non-unique axes)")
    # g.axes[3, 0].set_ylabel('Averaged cooling rate\n[K/year]')
    # g.axes[3, 0].invert_yaxis()

    g.axes[2, 0].set_xlabel('Volume [m$^3$]')
    g.axes[2, 0].set(xscale="log")
    g.axes[2, 1].set_xlabel('Average mantle temperature\n[K, at t = 0 years]')
    g.axes[2, 2].set_xlabel('Initial intrusion temperature\n[K, at t = 0 years]')
    # g.axes[3, 3].set_xlabel('Metal fraction of int. [by vol.]')
    g.axes[2, 0].set(yscale="log")

    for x in range(0, 3):
        for y in range(0, 3):
            g.axes[y, x].scatter(x=df2[x_variables[x]], y=df2[y_variables[y]], c='#CA054D', marker="o", s=70, edgecolor="white", alpha=0.6)


    g.axes[0, 0].axhline(y=zoning_constraint, c='#CA054D', ls="--")
    g.axes[0, 1].axhline(y=zoning_constraint, c='#CA054D', ls="--")
    g.axes[0, 2].axhline(y=zoning_constraint, c='#CA054D', ls="--")
    # g.axes[0, 3].axhline(y=zoning_constraint, c='#CA054D', ls="--")
    ## placement when not log plot:
    # text3 = g.axes[0, 0].text(1.6E7, 99, "> 95 %", fontsize=12,
    #                  c='#CA054D',)
    text3 = g.axes[0, 0].text(0.5E4, zoning_constraint+4, f"$\geq$ {int(zoning_constraint)} %", fontsize=11,
                    c='#CA054D',)
    text3.set_bbox(dict(facecolor='#A4D4B4', alpha=0.5, edgecolor='#CA054D'))

    # g.axes[1, 0].axhline(y=36, c='#CA054D', ls="--")
    g.axes[1, 0].axhline(y=rounding_constraint, c='#CA054D', ls="--")

    # g.axes[1, 1].axhline(y=36, c='#CA054D', ls="--")
    g.axes[1, 1].axhline(y=rounding_constraint, c='#CA054D', ls="--")

    # # g.axes[1, 2].axhline(y=36, c='#CA054D', ls="--")
    g.axes[1, 2].axhline(y=rounding_constraint, c='#CA054D', ls="--")

    # # g.axes[1, 3].axhline(y=36, c='#CA054D', ls="--")
    # g.axes[1, 3].axhline(y=rounding_constraint, c='#CA054D', ls="--")

    ## placement without log log plot
    # text4 = g.axes[1, 0].text(1.6E7, 65, "> 60 %", fontsize=12,
    #                  c='#CA054D',)
    text4 = g.axes[1, 0].text(0.5E4, rounding_constraint+5, f"$\geq$ {int(rounding_constraint)} %", fontsize=11,
                    c='#CA054D',)
    text4.set_bbox(dict(facecolor='#A4D4B4', alpha=0.5, edgecolor='#CA054D'))

    if len(passing_models) > 0:
        g.axes[0, 1].axvline(x=passing_models.avg_mantle_temp[0], c='#CA054D', ls=":")
        g.axes[1, 1].axvline(x=passing_models.avg_mantle_temp[0], c='#CA054D', ls=":")
        g.axes[2, 1].axvline(x=passing_models.avg_mantle_temp[0], c='#CA054D', ls=":")
    # g.axes[3, 1].axvline(x=passing_models.avg_mantle_temp[0], c='#CA054D', ls=":")

    g._legend.set_title(None)


    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles[1:5]+handles[6:8],
            labels=labels[1:5]+labels[6:8],
            ncol=3, loc="lower center", bbox_to_anchor=(0.4, 1.005), bbox_transform=plt.gcf().transFigure)

    # sns.move_legend(g, loc="lower left", bbox_to_anchor=(0.001, 1.035), ncol=3)

    g._legend.remove()
    return g


def kde_plot(df, zoning_constraint, rounding_constraint):
    f, ax = plt.subplots(ncols=1, nrows=3, constrained_layout=True, figsize=(3.5, 9))



    geom_params = ["int_temp", 'ext_temp', 'actual_volume', 'avg_r', ]
    pal = ['#1B98E0','#8A45C6','#e66101', '#CA054D', ]

    # top row
    # ax1 = ax[0]
    ax2 = ax[0]

    ax3 = ax[1]
    ax4 = ax[2]


    # ax2
    param = geom_params[1]
    i = sns.kdeplot(data=df, x=param, hue="Criteria_Matches", palette=pal, alpha=0.7, fill=False,  legend=False, ax=ax2, hue_order=["No match", "Rounding matches", "Zoning matches", "Both match"],)

    # ax3
    ax3.set_xscale("log")
    param = geom_params[2]
    i = sns.kdeplot(data=df, x=param, hue="Criteria_Matches", palette=pal, alpha=0.7, fill=False, legend=False, ax=ax3, hue_order=["No match", "Rounding matches", "Zoning matches", "Both match"],)

    # ax4
    param = geom_params[3]
    i = sns.kdeplot(data=df, x=param, hue="Criteria_Matches", palette=pal, alpha=0.7, fill=False, legend=False, ax=ax4, hue_order=["No match", "Rounding matches", "Zoning matches", "Both match"],)

    # setting up lines for legend

    legend_elements = [Line2D([0], [0], color=pal[0], alpha=0.7, ls="-", label="No match"),
                    Line2D([0], [0], color=pal[1], alpha=0.7, ls="-", label="Rounding matches"),
                    Line2D([0], [0], color=pal[2], alpha=0.7, ls="-", label="Zoning matches"),
                    Line2D([0], [0], color=pal[3], alpha=0.7, ls="-", label="Both match"),]

    # turning off spines


    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)

    ax4.spines["right"].set_visible(False)
    ax4.spines["top"].set_visible(False)

    # give axes titles/labels
    # ax1.set_xlabel("Initial intrusion\ntemperature [K]")
    ax2.set_xlabel("Mantle background\ntemperature [K]")
    # ax2.set_ylabel(None)

    ax3.set_xlabel("Intrusion volume\n[m$^3$]")
    # ax3.set_ylabel(None)

    ax4.set_xlabel("Average intrusion\nradius [m]")
    # ax4.set_ylabel(None)


    # ax1.ticklabel_format(style='scientific', scilimits=(0,0), axis='y', useMathText=True)
    ax2.ticklabel_format(style='scientific', scilimits=(0,0), axis='y', useMathText=True)
    ax4.ticklabel_format(style='scientific', scilimits=(0,0), axis='y', useMathText=True)

    f.legend(handles=legend_elements, ncol=2,  loc='center', bbox_to_anchor=(0.5, 1.05), frameon=False,
            title=None)


    return f



st.write(f'##### {len(passing_models)} models meet these criteria out of {len(data_calculations)} model runs ({round(100* len(passing_models)/len(data_calculations))} % of models).')

col1, col2 = st.columns([0.7, 0.3])

with col1:
    fig = plotting_values(zoning_cons,
                        rounding_cons,
                        data_calculations,
                        passing_models,
                        df, df2, df_plan)

    st.pyplot(fig)
    st.write("Vertical dashed line on temperature plots shows highest mantle temperature that meets both criteria.")

with col2:
    fig2 = kde_plot(df,
                    zoning_cons,
                    rounding_cons)



    st.pyplot(fig2)

# sidebar with info


sidebar_content0 = """
Pallasite meteorites are beautiful mixture of iron-nickle metal and green olivine crystals.
They formed 4.6 billion years ago in the chaotic early Solar System, through repeated collisions
between small planetary bodies called **planetesimals**. Have a look at this [digital poster](https://www.panorama.researchposter.co.uk/2020/10/09/murphy-quinlan-maeve-peek-inside-a-planetesimal/)
to find out more about planeteismals.

This page shows the results of [Murphy Quinlan et al. (2023)](https://doi.org/10.1016/j.epsl.2023.118284), which investigated the seemingly
contradictory cooling rates suggested by the chemistry and morphology of pallasite olivine
crystals.

The suite of models shown here demonstrate the effect of parameters such as the temperature
of the planetesimal mantle, the geometry of the intrusion region (where molten iron-nickle
and crystalline olivine mixed), and the aspect ratio of the intrusion region have on either
geochemical zoning preservation or facilitation of macroscale rounding of olivine crystals.
"""


sidebar_content1 = """
### Paper highlights:

- Fast and slow cooling recorded in pallasites can be explained by metal intrusion.
- Small intrusions with pipe or sheet morphology favour olivine zoning preservation.
- No ad-hoc changes to the parent body required for contrasting cooling rates.
"""

abstract = """
See the [visual abstract here](https://doi.org/10.1016/j.epsl.2023.118284).

Pallasite meteorites contain evidence for vastly different cooling timescales:
rapid cooling at high temperatures (K/yrs) and slow cooling at lower temperatures
(K/Myrs). Pallasite olivine also shows contrasting textures ranging from well-rounded
to angular and fragmental, and some samples record chemical zoning. Previous pallasite
formation models have required fortuitous changes to the parent body in order to explain
these contrasting timescales and textures, including late addition of a megaregolith
layer, impact excavation, or parent body break-up and recombination.
We investigate the timescales recorded in Main Group Pallasite meteorites with a
coupled multiscale thermal diffusion modelling approach, using a 1D model of the parent
body and a 3D model of the metal-olivine intrusion region, to see if these large-scale
changes to the parent body are necessary. We test a range of intrusion volumes and aspect
ratios, metal-to-olivine ratios, and initial temperatures for both the background mantle
and the intruded metal. We find that the contrasting timescales, textural heterogeneity,
and preservation of chemical zoning can all occur within one simple ellipsoidal segment
of an intrusion complex. These conditions are satisfied in 13% of our randomly generated
models (2200 model runs), with small intrusion volumes (with a mean radius ≲100 m) and colder
background mantle temperatures (≲1200 K) favourable. Large rounded olivine can be
explained by a previous intrusion of metal into a hotter mantle, suggesting possible
repeated bombardment of the parent body. We speculate that the formation of pallasitic
zones within planetesimals may have been a common occurrence in the early Solar System,
as our model shows that favourable pallasite conditions can be accommodated in a wide
range of intrusion morphologies, across a wide range of planetesimal mantle temperatures,
without the need for large-scale changes to the parent body. We suggest that pallasites
represent a late stage of repeated injection of metal into a cooling planetesimal mantle,
and that heterogeneity observed in micro-scale rounding or chemical zoning preservation
in pallasite olivine can be explained by diverse cooling rates in different regions of
a small intrusion.
"""
sidebar_content2="""
- **[Read the full paper in Earth and Planetary Science Letters](https://doi.org/10.1016/j.epsl.2023.118284)**
- Model your own planetesimals with [Pytesimal](https://murphyqm.github.io/pytesimal/)
- Model metallic intrusion in planetesimals with [PytesiMINT](https://github.com/murphyqm/pytesimint)
"""

with st.sidebar:
    st.header("Reconciling fast and slow cooling during planetary formation as recorded in the main group pallasites")
    st.subheader("Maeve Murphy Quinlan, Andrew M. Walker, Chris J. Davies, 2023.")
    with st.expander("What are pallasites?"):
        st.write(sidebar_content0)
    st.markdown(sidebar_content1)
    with st.expander("Read the abstract"):
        st.write(abstract)
    st.markdown(sidebar_content2)


