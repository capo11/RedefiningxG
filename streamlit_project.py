import streamlit as st
import pandas as pd
from mplsoccer import VerticalPitch
import joblib
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import xgboost
import shap

shotsMultiplier = 500

pd.options.mode.chained_assignment = None
# st.set_option('deprecation.showPyplotGlobalUse', False)

def cleanDataset(df, elo=False):
  #print("Cleaning: Preprocessing...")
  df = df.dropna(subset=['xg'])
  df = df.loc[df['position'] != 'G']
  # df = df[df.situation != "penalty"]
  x = pd.get_dummies(df, columns=['position'], prefix='position', drop_first=False)
  x = pd.get_dummies(x, columns=['situation'], prefix='situation', drop_first=False)
  x = pd.get_dummies(x, columns=['bodyPart'], prefix='bodyPart', drop_first=False)
  #print("Cleaning: Scaling...")
  scaler = MinMaxScaler()
  # x[['minute']] = scaler.fit_transform(x[['minute']])

  # x[['rating']] = scaler.fit_transform(x[['rating']])
  #print("Normalizing Rating")
  x['rating'] = x['rating']/100
  # x[['eloTeam']] = scaler.fit_transform(x[['eloTeam']])
  #print("Normalizing Elo Team")
  if elo == True:
    x['eloTeam'] = x['eloTeam']/2168
  # x[['keeperRating']] = scaler.fit_transform(x[['keeperRating']])
  #print("Normalizing Keeper Rating")
  #print(x['keeperRating'])
  x['keeperRating'] = (x['keeperRating'].astype(int))/100
  # x[['eloOpponent']] = scaler.fit_transform(x[['eloOpponent']])
  #print("Normalizing Elo Opponent")
  if elo == True:
    x['eloOpponent'] = x['eloOpponent']/2168
  # x[['distance']] = scaler.fit_transform(x[['distance']])
  x['distance'] = x['distance']/120
  # x[['angle']] = scaler.fit_transform(x[['angle']])
  x['angle'] = x['angle']/90
  # x['distAngle'] = (0.75 * x['distance']) + (0.25 * x['angle'])
  #print("Cleaning: Converting...")
  x['position_D'] = x['position_D'].astype(int)
  x['position_F'] = x['position_F'].astype(int)
  # x['position_G'] = x['position_G'].astype(int)
  x['position_M'] = x['position_M'].astype(int)

  x['situation_assisted'] = x['situation_assisted'].astype(int)
  x['situation_corner'] = x['situation_corner'].astype(int)
  x['situation_fast-break'] = x['situation_fast-break'].astype(int)
  x['situation_free-kick'] = x['situation_free-kick'].astype(int)
  # print(x.columns)
  # print(np.unique(df['situation']))
  x['situation_penalty'] = x['situation_penalty'].astype(int)
  x['situation_regular'] = x['situation_regular'].astype(int)
  x['situation_set-piece'] = x['situation_set-piece'].astype(int)
  x['situation_throw-in-set-piece'] = x['situation_throw-in-set-piece'].astype(int)

  x['bodyPart_head'] = x['bodyPart_head'].astype(int)
  x['bodyPart_left-foot'] = x['bodyPart_left-foot'].astype(int)
  x['bodyPart_other'] = x['bodyPart_other'].astype(int)
  x['bodyPart_right-foot'] = x['bodyPart_right-foot'].astype(int)
  #print("Cleaning: Separating X and Y...")
  y = df['goal']
  # print("X Columns")
  # print(x.columns)
  x = x.drop(columns=['goal', 'player', 'team', 'keeper', 'opponent', 'isHome', 'x', 'y', 'xg'])
  # x = x.drop(columns=['goal', 'player', 'team', 'keeper', 'opponent', 'isHome', 'x', 'y', 'xg', 'distance', 'angle'])
  return x, y

def getXTrain(df, elo=False, over='none', k=15, sampling_strategy='none', test_size=0.2):
    df = df.dropna(subset=['xg'])
    x,y = cleanDataset(df, elo)
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=test_size, random_state=42)
    if(over == "random"):
        oversample = RandomOverSampler(sampling_strategy=0.25, random_state=42)
        x,y = oversample.fit_resample(x,y)
    elif(over == "smote"):
        if(k!=0):
            smt = SMOTE(k_neighbors=k, random_state=42)
        else:
            if(sampling_strategy=='none'):
                smt = SMOTE(random_state=42)
            else:
                smt = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        x,y = smt.fit_resample(x,y)
    elif(over == "adasyn"):
        ada = ADASYN(random_state=42)
        x,y = ada.fit_resample(x,y)
    
    
    return X_train, x

def predictLocalGame(game, model):
    allShots = pd.read_csv('seriea2425.csv')
    shotmap = allShots.loc[(allShots['homeTeam'] == game['home_team']) & (allShots['awayTeam'] == game['away_team'])]
    shotmap = shotmap.reset_index()
    shotmap = shotmap.drop(columns=['level_0'])
    shotmap = shotmap.drop(columns=['minute', 'index', 'round', 'homeTeam', 'awayTeam'])
    shotmap = shotmap.dropna(subset=['xg'])
    shotmap = shotmap.loc[shotmap['position'] != 'G']
    shotmap = shotmap.drop(columns=['eloTeam', 'eloOpponent'])

    homeShots = shotmap.loc[shotmap['isHome'] == True]
    homeShots = homeShots.reset_index()
    homeShots = homeShots.drop(columns=['index'])
    awayShots = shotmap.loc[shotmap['isHome'] == False]
    awayShots = awayShots.reset_index()
    awayShots = awayShots.drop(columns=['index'])

    df = pd.read_csv('seriea_joined.csv')
    df = df.drop(columns=['minute', 'eloTeam', 'eloOpponent'])
    df_homeShots = pd.concat([df, homeShots]).reset_index()
    df_x, df_y = cleanDataset(df_homeShots)

    homeShots_clean = df_x.loc[len(df):]
    homeShots_clean = homeShots_clean.drop(columns=['index'])
    homeShots_clean = homeShots_clean.reset_index()
    homeShots_clean = homeShots_clean.drop(columns=['index'])
    homeXgPred = model.predict_proba(homeShots_clean)[:, 1]
    homePred = model.predict(homeShots_clean)
    homeShots['goalPred'] = homePred
    homeShots['xgPred'] = homeXgPred
    for i in homeShots.index:
        if (homeShots.loc[i]['situation'] == 'penalty'):
            homeShots.at[i, 'xgPred'] = 0.75
        if (homeShots.loc[i]['xgPred'] == 0):
            homeShots.at[i, 'xgPred'] = 0.01
    homeShots['diff'] = homeShots['xgPred']-homeShots['xg']

    df_awayShots = pd.concat([df, awayShots]).reset_index()
    df_x, df_y = cleanDataset(df_awayShots)
    awayShots_clean = df_x.loc[len(df):]
    awayShots_clean = awayShots_clean.drop(columns=['index'])
    awayShots_clean = awayShots_clean.reset_index()
    awayShots_clean = awayShots_clean.drop(columns=['index'])
    awayXgPred = model.predict_proba(awayShots_clean)[:, 1]
    awayPred = model.predict(awayShots_clean)
    awayShots['goalPred'] = awayPred
    awayShots['xgPred'] = awayXgPred
    for i in awayShots.index:
        if (awayShots.loc[i]['situation'] == 'penalty'):
            awayShots.at[i, 'xgPred'] = 0.75
        if (awayShots.loc[i]['xgPred'] == 0):
            awayShots.at[i, 'xgPred'] = 0.01
    awayShots['diff'] = awayShots['xgPred']-awayShots['xg']

    stats = {
      "shotmap": shotmap,
      "homeShots": homeShots,
      "awayShots": awayShots,
      "homeShots_clean": homeShots_clean,
      "awayShots_clean": awayShots_clean,
      "homeXgPred": homeXgPred,
      "homePred": homePred,
      "awayXgPred": awayXgPred,
      "awayPred": awayPred
    }
    stats['homeShots']['xg'] = round(stats['homeShots']['xg'], 2)
    stats['homeShots']['xgPred'] = round(stats['homeShots']['xgPred'], 2)
    stats['homeShots']['diff'] = round(stats['homeShots']['diff'], 2)
    stats['awayShots']['xg'] = round(stats['awayShots']['xg'], 2)
    stats['awayShots']['xgPred'] = round(stats['awayShots']['xgPred'], 2)
    stats['awayShots']['diff'] = round(stats['awayShots']['diff'], 2)
    return stats

def plotShap(shapValues):
    features = []
    shap_values = []
    values = []
    for (i, feature) in enumerate(shapValues.data.index):
        print(i,feature)
        features.append(feature)
        shap_values.append(round(shapValues.values[i], 2))
    for (i, value) in enumerate(shapValues.data):
        # print(i, value)
        if (i==0):
            value = int(value)  #differenza goal
        elif(i==1 or i==2):
            value = int(value*100)  #ratings
        elif(i==3):
            value = value*120   #distanza
        elif(i==4):
            value = value*90    #angolo
        else:
            value = int(value)
        values.append(value)
    
    # print(features)
    # print(values)
    features = ['Differenza Goal:', 'Rating Tiratore:', 'Rating Portiere:', 'Distanza:', 'Angolo:', 'Posizione - Difensore', 'Posizione - Attaccante', 'Posizione - Centrocampista', 'Situazione - Servito', 'Situazione - Corner', 'Situazione - Contropiede', 'Situazione - Punizione', 'Situazione - Rigore', 'Situazione - Regolare', 'Situazione - Calcio Piazzato', 'Situazione - Rimessa Laterale', 'Corpo - Testa', 'Corpo - Piede Sinistro', 'Corpo - Altro', 'Corpo - Piede Destro']
    features_values = []
    for i in range(0, len(features)):
        if(i<=4):
            features_values.append(str(features[i]) + ' ' + str(round(values[i], 2)))
        else:
            if(values[i] == 1):
                features_values.append(str(features[i]) + ': Si')
            else:
                features_values.append(str(features[i]) + ': No')
    print(features_values)

    shap_values = np.array(shap_values)
    sorted_indices = np.argsort(np.abs(shap_values))[10:]
    sorted_shap_values = shap_values[sorted_indices]
    sorted_feature_names = [features_values[i] for i in sorted_indices]
    plt.figure(figsize=(8, 6))
    bars = plt.barh(sorted_feature_names, sorted_shap_values, color=["green" if v > 0 else "red" for v in sorted_shap_values])
    plt.xlabel("Shapley Value", color="white")
    plt.ylabel("Feature", color="white")
    plt.title("Quali fattori influenzano il tiro?", color="white")
    plt.axvline(0, color="white", linewidth=0.8, linestyle="--")  # Linea verticale per il riferimento a zero
    # plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tick_params(axis='both', colors='white')
    for pos in ['right', 'top', 'bottom', 'left']: 
        plt.gca().spines[pos].set_visible(False) 


    # Mostrare il grafico
    # plt.show()
    st.pyplot(transparent=True)


st.title("Serie A 2024/25")
st.subheader("Filtra per Partita e Tiro per vedere la mappa dei tiri e le differenze di xG!")


modelDescription = st.selectbox('Seleziona il Modello', ['Random Forest', 'XGBoost'], index=1)
if modelDescription:
    if modelDescription == 'Random Forest':
        modelName = 'base_FOR'
    elif modelDescription == 'XGBoost':
        modelName = 'base_XGB'

# modelName = "SMOTE_SS0.5"
# modelName = "SMOTE_K15"
# modelName = "ADASYN"
    model = joblib.load('models/' + modelName + '.sav')
    df = pd.read_csv('seriea_joined.csv')
    df = df.drop(columns=['minute', 'eloTeam', 'eloOpponent'])
    X_train, X = getXTrain(df)
    explainer = shap.Explainer(model, X_train)
    # explainer = shap.Explainer(model, X)



    

    shotsDF = pd.read_excel('allShots/allShots_' + modelName + '.xlsx')
    shotsDF = shotsDF.drop(columns='Unnamed: 0')
    # print(shotsDF.head())
    statsDF = pd.read_excel('leagueStats/leagueStats_' + modelName + '.xlsx')
    statsDF = statsDF.drop(columns='Unnamed: 0')
    # print(statsDF.head())


    schedule = pd.read_csv('serieaSchedule.csv')
    schedule = schedule.drop(columns='Unnamed: 0')
    scheduleDone = schedule.loc[schedule['week']<=13]
    descriptions = []
    for i in scheduleDone.index:
        description = str(scheduleDone.loc[i]['week']) + '° Giornata: ' + scheduleDone.loc[i]['home_team'] + ' - ' + scheduleDone.loc[i]['away_team'] + ' ' + str(int(scheduleDone.loc[i]['home_score'])) + ' - ' + str(int(scheduleDone.loc[i]['away_score']))
        descriptions.append(description)
    scheduleDone['description'] = descriptions
    # print(scheduleDone.head())



    gameDescription = st.selectbox('Seleziona una Partita', scheduleDone['description'], index=None)
    if gameDescription:
        gameIndex = scheduleDone.loc[scheduleDone['description'] == gameDescription].index[0]
        st.error("xG Sofascore: " + str(statsDF.loc[gameIndex]['homeXg']) + ' - ' + str(statsDF.loc[gameIndex]['awayXg']))
        st.info("xG Previsti dal Modello: " + str(statsDF.loc[gameIndex]['homeXgPred']) + ' - ' + str(statsDF.loc[gameIndex]['awayXgPred']))
        # print(gameIndex)
        stats = predictLocalGame(scheduleDone.loc[gameIndex], model)
        gameShots = shotsDF.loc[shotsDF['gameIndex'] == gameIndex]
        # print(gameShots.head())
        pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#22312b', half=True)
        fig,axs = pitch.draw(figsize=(8,4), ncols=2)
        fig.set_facecolor('#22312b')
        axs[0].patch.set_facecolor('#22312b')
        axs[0].set_title("xG Sofascore", color="white")
        axs[1].patch.set_facecolor('#22312b')
        axs[1].set_title("xG Modello", color="white")
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='xG Modello > xG Sofascore'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='xG Modello = xG Sofascore'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='xG Modello < xG Sofascore')
        ]

        axs[1].legend(handles=legend_elements, loc='lower center', title='Legenda')
        
        home_team = scheduleDone.loc[gameIndex]['home_team']
        away_team = scheduleDone.loc[gameIndex]['away_team']

        selectedTeam = st.selectbox('Seleziona una Squadra', [home_team, away_team], index=None)
        # print(home_team, away_team)

        if selectedTeam:
            teamShots = gameShots.loc[gameShots['team'] == selectedTeam].reset_index(drop=True)
            # print(teamShots.head())
            descriptions = []
            for i in teamShots.index:
                if(teamShots.loc[i]['goal'] == 0):
                    shotOutcome = 'No Goal'
                else:
                    shotOutcome = 'Goal'
                description = str(i+1) + ' - ' + teamShots.loc[i]['player'] + ' - ' + shotOutcome + ' (' + str(teamShots.loc[i]['xg']) + ' xG)'
                # print(shotDescription)
                descriptions.append(description)
                x = 120-teamShots.loc[i]['x']
                y = teamShots.loc[i]['y']
                pitch.scatter(
                    x=x, 
                    y=y,
                    ax=axs[0],
                    s = shotsMultiplier*teamShots.loc[i]['xg'],
                    c='black',
                    edgecolors='white')
                if(teamShots.loc[i]['xg']<teamShots.loc[i]['xgPred']):
                    color='green'
                elif(teamShots.loc[i]['xg']>teamShots.loc[i]['xgPred']):
                    color='red'
                else:
                    color='yellow'
                pitch.scatter(
                    x=x, 
                    y=y,
                    ax=axs[1],
                    s = shotsMultiplier*teamShots.loc[i]['xgPred'],
                    c=color,
                    edgecolors='white')
            # fig_html = mpld3.fig_to_html(fig)
            # size = fig.get_size_inches()
            # width = int(size[0]) * 100+50
            # height = int(size[1]) * 100+50
            # components.html(fig_html, height=height, width=width)
            st.pyplot(fig)
        
            teamShots['description'] = descriptions
            shotDescription = st.selectbox('Seleziona un Tiro', teamShots['description'], index=None)
            if shotDescription:
                shotIndex = teamShots.loc[teamShots['description'] == shotDescription].index[0]
                # print(shotIndex)

                # pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#22312b', half=True)
                # fig,ax = pitch.draw(figsize=(8,4))
                # fig.set_facecolor('#22312b')
                # ax.patch.set_facecolor('#22312b')
                # for i in teamShots.index:
                #     x = 120-teamShots.loc[i]['x']
                #     y = teamShots.loc[i]['y']
                #     if(i == shotIndex):
                #         xg = teamShots.loc[i]['xg']
                #         xgPred = teamShots.loc[i]['xgPred']
                #         if(xg>xgPred):
                #             pitch.scatter(
                #             x=x, 
                #             y=y,
                #             ax=ax,
                #             s = shotsMultiplier*xg,
                #             c='red',
                #             edgecolors='white')
                #             pitch.scatter(
                #                 x=x, 
                #                 y=y,
                #                 ax=ax,
                #                 s = shotsMultiplier*xgPred,
                #                 c='blue',
                #                 edgecolors='white')
                #         else:
                #             pitch.scatter(
                #                 x=x, 
                #                 y=y,
                #                 ax=ax,
                #                 s = shotsMultiplier*xgPred,
                #                 c='blue',
                #                 edgecolors='white')
                #             pitch.scatter(
                #                 x=x, 
                #                 y=y,
                #                 ax=ax,
                #                 s = shotsMultiplier*xg,
                #                 c='blue',
                #                 edgecolors='white')
                # st.pyplot(fig)
                
                st.error("xG Sofascore: " + str(teamShots.loc[shotIndex]['xg']))
                st.info("xG Previsto dal Modello: " + str(teamShots.loc[shotIndex]['xgPred']))
                
                if(selectedTeam == home_team):
                    shot = stats['homeShots_clean'].loc[shotIndex]
                elif(selectedTeam == away_team):
                    shot = stats['awayShots_clean'].loc[shotIndex]
                
                shapValues = explainer(shot)
                # st.pyplot(shap.plots.force(shapValues, matplotlib=True))
                plotShap(shapValues)
                    
            