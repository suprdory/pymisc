# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import sys
# sys.argv
# %%
df0 = pd.read_csv("ar6_snapshot_1726223253.csv")

# %% select region
df0 = df0[df0["Region"] == "World"]
# %%
sorted(list(df0.Variable.unique()))

# %%

varDict = {
    'PR': "Primary Energy|Non-Biomass Renewables",
    'PB': "Primary Energy|Biomass",
    'PT': "Primary Energy",
    'SR': "Secondary Energy|Electricity|Non-Biomass Renewables",
    'SB': "Secondary Energy|Electricity|Biomass",
    'ST': "Secondary Energy|Electricity",
    'EC': "Emissions|CO2"
}


# %%
models = df0['Model'].unique()
scens = df0['Scenario'].unique()
years = [2020, 2025, 2030, 2035, 2040, 2045, 2050]
print("n mods:", len(models))
print("n scenes:", len(scens))

# %% build main dataframe by extracting each var in varDict by model, scenario, and year.


def get_var(dfi, varname, year):
    dfv = dfi[dfi['Variable'] == varname]
    if len(dfv) > 1:
        raise Exception("Multiple values found")
    if len(dfv) == 0:
        dfvy = np.nan
    elif len(dfv) == 1:
        dfvy = dfv[year].values[0]
    return dfvy


rows = []
for model in models:
    # print("Model:", model)
    for scen in scens:
        # print("Scenario:", scen)
        dfi = df0[(df0["Model"] == model) & (df0["Scenario"] == scen)]
        for year in years:
            if len(dfi) > 0:
                row = {'mod': model,
                       'scen': scen,
                       'year': year}
                for var in varDict:
                    row[var] = get_var(dfi, varDict[var], str(year))
                rows.append(row)
df = pd.DataFrame(rows)

df["PRF"] = (df.PR+df.PB)/df.PT
df["SRF"] = (df.SR + df.SB) / df.ST
varDict['PRF'] = "Primary Renewable Fraction"
varDict['SRF'] = "Secondary Renewable Fraction"
df=df.drop(columns=['PR', 'PB', 'SR', 'SB'])

# %% convert to multindex 
df_ts = df.sort_values(['mod', 'scen', 'year']).set_index(
    ['mod', 'scen', 'year'])
# df_ts=df_ts.reset_index(level=["year"])
df_ts
#%%

def plot_ts(ax, df, yvar):
    cmap = plt.get_cmap('tab20b')
    cmap2 = plt.get_cmap('tab20c')
    for sx, group in enumerate(df.index.levels[0]):
        # print(sx)
        if sx<20:
            col=cmap(sx)
        else:
            col=cmap2(sx%20)
        yr = df.loc[group].index
        v = df.loc[group][yvar]
        ax.plot(yr, v, lw=2, label=group, color=col)
    ax.grid()
    ax.set_xlabel('Year')
    ax.set_ylabel(varDict[yvar])
yvars= ['PRF', 'SRF', 'EC']

#%% plot scenario means
df_ts_scen_mean = df.groupby(['scen', 'year']).mean(numeric_only=True)
fig, axs = plt.subplots(3, 1, dpi=200, figsize=(10, 10))
for ax, yvar in zip(axs, yvars):
    plot_ts(ax, df_ts_scen_mean, yvar)
ax.legend(bbox_to_anchor=(1.0, 3.7,))
axs[0].set_title("Scenario means")
# %% plot model means
df_ts_model_mean = df.groupby(['mod', 'year']).mean(numeric_only=True)
fig, axs= plt.subplots(3, 1, dpi=200, figsize=(10, 10))
for ax, yvar in zip(axs, yvars):
    plot_ts(ax, df_ts_model_mean, yvar)
ax.legend(bbox_to_anchor=(1.32, 2.7,))
axs[0].set_title("Model means")

# %% Scatter
dfy= df[(df.year == 2025)]
fig, ax= plt.subplots(1, 1)
xvar= 'EC'
yvar= 'PRF'
ax.plot(dfy[xvar], dfy[yvar], '.')
ax.grid()
ax.set_xlabel(varDict[xvar])
ax.set_ylabel(varDict[yvar])

vx= (np.isfinite(dfy[xvar])) & (np.isfinite(dfy[yvar]))
rho= np.corrcoef(dfy[xvar][vx], dfy[yvar][vx])[0, 1]
print(f'Corr Coef:{rho:0.2f}')

# %% Find stuff
df[(df.year==2050)&(df.SRF<0.5)]


# %%
